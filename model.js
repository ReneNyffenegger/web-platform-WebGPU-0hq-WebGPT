
function tq84_spaces(n) { return ' '.repeat(n*2); }
function tq84_dumpObjectStructure(obj, indent=0) {
// console.log(tq84_dumpObjectStructure( [1, 2, 3 ]))
// console.log(tq84_dumpObjectStructure( [1, ['a', 'b', 'c'], 3, [], {} ]))
// console.log(tq84_dumpObjectStructure( [1, 2, ['level', 'one', 'array'], {x: null, y: "foo bar baz", z:{ ary: ['x', 'y', navigator.gpu, 'z'], emptyObj: {} }} ]  ))

   if (Array.isArray(obj)) {
      return "[" +
         obj.map( item =>  "\n" + tq84_spaces(indent+1) + tq84_dumpObjectStructure(item, indent+1)).join(",") +
        "\n" + tq84_spaces(indent) + "]";
   }

   if (typeof obj === 'object' && obj !== null /* && obj.constructor.name == 'Object' */ ) {

     if (obj.constructor.name == 'Object') {

      return "{" +
        Object.keys(obj).map(
           k => "\n" + tq84_spaces(indent+1) + k + ': ' + tq84_dumpObjectStructure(obj[k], indent+1)
        ).join("," ) +
      "\n" + tq84_spaces(indent) + "}";

     }
     return '<' + obj.constructor.name + '>';

   }

   if (obj === null) {
      return 'null';
   }

   return '<' + typeof(obj) + '>';
}




class GPT {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;

    this.defaultPrompt;
    this.defaultTopK;
    this.defaultTemperature;
    this.defaultTokens;

    this.externalBuffer;

    this.unloadDeletionStack = [];
  }

  async initialize() {
    console.log('  initialize');
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    console.log(`   device = ${this.device.constructor.name}`);
    console.log(`   device.limits.maxStorageBufferBindingSize = ${this.device.limits.maxStorageBufferBindingSize} (${this.device.limits.maxStorageBufferBindingSize /1024/1024} MB)`);

    initializeOperations(this.device);

    this.weightsFolder = `weights/${this.folder}/`;
    await this.loadParameters();

//  [this.model, this.params] = await this.loadModel(this.folder);
    [this.model             ] = await this.loadModel(this.folder);

    this.tokenizer = this.tokenizerType == "bpe" ? new GPT2Tokenizer() : new SimpleTokenizer();
    console.log('   call this.tokenizer.load');
    await this.tokenizer.load();

    if (this.tokenizerType == "bpe") {
      console.log('   tokenizerType = bpe');
//    this.defaultPrompt      = `What is the answer to life, the universe, and everything?\n`;
      this.defaultPrompt      = "don't answer me.";
      this.defaultTopK        = 3;
//    this.defaultTemperature = 1;
      this.defaultTemperature = 0;
      this.defaultTokens      = 30;
    }
    else {
      console.log('   tokenizerType != bpe');
      this.defaultPrompt = `WILL:\nAh, how dare you challenge me?\nHave you forgotten I built WebGPT?\n`;
      this.defaultTopK = 2;
      this.defaultTemperature = 1;
//    this.defaultTokens = 80;
      this.defaultTokens =  5;
    }

    this.initialized = true;

    console.log("  Model initialized");
  }

  async *generate(prompt, max_new_tokens, top_k, temperature) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    console.log('  model.generate');
    console.log('   prompt = ', prompt);
    console.log('   max_new_tokens = ', max_new_tokens);
    console.log('   top_k = ', top_k);
    console.log('   temperature = ', temperature);

    // Buffer size (321644800) exceeds the max buffer size limit (268435456).
    //  - While calling [Device].CreateBuffer([BufferDescriptor]).

    let history = this.tokenizer.encode(prompt);
//  console.log(`Prompt (${history.length} tokens):\n${prompt}`);
    console.log('   history = ', history);

    const warmupRuns = 3;
    let totalTime = 0;

    for (let i = 0; i < max_new_tokens; i++) {
    console.log('   i = ', i);
    //
    //  Get the lst n_ctx tokens
    //
       const idx_cond = history.slice(-this.params.n_ctx);

       const startTime = performance.now();
       const logits    = await this.run(idx_cond);
       const endTime   = performance.now();

      // console.log(`\nIteration ${i + 1} of ${max_new_tokens}`);
       const lapsedTime = endTime - startTime;
//     console.log(`Kernel execution time: ${lapsedTime} ms`);
       i >= warmupRuns && (totalTime += lapsedTime);

       const { topKIndices, topKProbs } = selectTopK(logits, top_k);
       const probs = cpuSoftmax(topKProbs, temperature);
       const idx_next = topKIndices[sampleFromDistribution(probs)];

       history = history.concat(idx_next);

      // console.log(`Output:\n${this.tokenizer.decode(history)}`);

      // const totalProbs = cpuSoftmax(logits, temperature);
      // const tokenProbsString = Array.from(totalProbs)
      //   .map((value, index) => ({ value, index }))
      //   .sort((a, b) => b.value - a.value)
      //   .slice(0, 8)
      //   .map((prob) => `{ ${this.tokenizer.decode([prob.index]).replace(/(\r\n|\n|\r)/gm, "newline")} } : ${prob.value.toPrecision(3)}`)
      //   .join(" | ");
      // console.log("Top 8 token probs:", tokenProbsString);

       yield this.tokenizer.decode([idx_next]);
    }

    console.log(`Average kernel execution time: ${totalTime / (max_new_tokens - warmupRuns)} ms`);
  }

  async run(idx) {
    console.log('    model.run');

    // ---------------- Create Passes ---------------- //
    // Note: These are re-initialized because everytime idx.length changes buffers are different sizes.

    // Pipeline creation is major bottleneck to spin up speed! Also buffer re-use.

    this.computePasses = [];
    let intermediateBuffer;
    let residualBuffer;

    {
      const { passes, resultBuffer } = EmbedBlock.newInstance(idx, this.params.n_embd, this.params.vocab_chunk_size, this.embeddingsBuffers, this.model.posEmbdBuffer);

      console.log('flags: '     + passes.map( (p) => p.flag      ).join(', '));
      console.log('srcOffset: ' + passes.map( (p) => p.srcOffset ).join(', '));
      console.log('dstOffset: ' + passes.map( (p) => p.dstOffset ).join(', '));
      console.log('size:      ' + passes.map( (p) => p.size      ).join(', '));

//    console.log('     passes = ');
//    console.log(tq84_dumpObjectStructure(passes));
//    The structure of passes is
//
//      {
//        flag: <string>
//        src: <GPUBuffer>,
//        srcOffset: <number>,
//        dst: <GPUBuffer>,
//        dstOffset: <number>,
//        size: <number>
//      },
//

      console.log('     resultBuffer = ');
      console.log(tq84_dumpObjectStructure(resultBuffer));

      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }


    for (let layer = 0; layer < this.params.n_layer; layer++) {

//    const buffers = this.model.layer_buffers[layer];
      const buffers =       this.layer_buffers[layer];

      {
        const { passes, resultBuffer } = LayerNormBlock.newInstance(
          idx.length,
          this.params.n_embd,
          intermediateBuffer,
          buffers.normAttentionGammaBuffer,
          buffers.normAttentionBetaBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }

      {
        const { passes, resultBuffer } = AttentionBlock.newFusedInstance(
          idx.length,
          this.params.n_embd,
          this.params.attention_scale,
          this.params.n_head,
          this.params.head_size,
          intermediateBuffer,
          buffers.qkvWeightArray[0], buffers.qkvBiasArray[0],
          buffers.qkvWeightArray[1], buffers.qkvBiasArray[1],
          buffers.qkvWeightArray[2], buffers.qkvBiasArray[2],
          buffers.linearWeightsBuffer,
          buffers.linearBiasBuffer,
          FastMatMulBlock,
          SoftmaxBlock
        );

        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }

      {
        const { passes, resultBuffer } = ResidualBlock.newInstance(idx.length, this.params.n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {
        const { passes, resultBuffer } = LayerNormBlock.newInstance(
          idx.length,
          this.params.n_embd,
          intermediateBuffer,
          buffers.normLinearGammaBuffer,
          buffers.normLinearBetaBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }

      {
        const { resultBuffer, passes } = FastMatMulBlock.newInstance(
          idx.length,
          this.params.hidden_size,
          this.params.n_embd,
          intermediateBuffer,
          buffers.firstLayerWeightsBuffer,
          buffers.firstLayerBiasBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
      {

        const { resultBuffer, passes } = GeluBlock.newInstance(idx.length, this.params.hidden_size, intermediateBuffer);
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }

      {
        const { resultBuffer, passes } = FastMatMulBlock.newInstance(
          idx.length,
          this.params.n_embd,
          this.params.hidden_size,
          intermediateBuffer,
          buffers.secondLayerWeightsBuffer,
          buffers.secondLayerBiasBuffer
        );
        intermediateBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }

      {
        const { passes, resultBuffer } = ResidualBlock.newInstance(idx.length, this.params.n_embd, intermediateBuffer, residualBuffer);
        intermediateBuffer = resultBuffer;
        residualBuffer = resultBuffer;
        this.computePasses.push(...passes);
      }
    }

    {
      if (this.externalBuffer) {
        this.computePasses.push({
          flag: "copy",
          src: intermediateBuffer,
          srcOffset: 0,
          dst: this.externalBuffer,
          dstOffset: 0,
          size: this.bufferSize(idx.length, this.params.n_embd),
        });
      }
    }

    {
      const { passes, resultBuffer } = LayerNormBlock.newInstance(idx.length, this.params.n_embd, intermediateBuffer, this.model.normGammaBuffer, this.model.normBetaBuffer);
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    {
      const { passes, resultBuffer } = DeEmbedBlock.newInstance(
        this.params.n_embd,
        this.params.vocab_size,
        this.params.vocab_chunk_size * this.params.vocab_chunk_instances,
        idx.length,
        this.params.vocab_chunk_size,
        intermediateBuffer,
//      this.model.deEmbeddingsBuffers
        this.      deEmbeddingsBuffers
      );
      intermediateBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }
    const resultBuffer = intermediateBuffer;

 // ---------------- Compute Passes ----------------

    console.log('     model.run - Compute Passes');

//
//  Create a GPUCommandEncoder instance:
    const commandEncoder = this.device.createCommandEncoder();
//  console.log(`     commandEncoder = ${commandEncoder.constructor.name}`);

    for (const pass of this.computePasses) {
       if (pass.flag === "compute") {
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pass.pipeline);

          for (let i = 0; i < pass.groups.length; i++) {
             passEncoder.setBindGroup(i, pass.groups[i]);
          }

          passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y);
          passEncoder.end();
        }
        else if (pass.flag === "copy") {
          commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
       }
    }

    this.device.queue.submit([commandEncoder.finish()]);

 // ---------------- Read Results ----------------

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const output = resultBuffer.getMappedRange();
    const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.

    clearOperationCache();

    return outputArray;
  }

  async loadModel(folder) {

    console.log(`    loadModel folder=${folder}`);
    if (this.initialized) return console.error("Model already loaded");

//  console.log("Loading model from folder:", folder);

    await this.loadEmbeddings();

//  console.log(`     loadEmbeddings returned embeddingsBuffers.length = ${embeddingsBuffers.length} (${embeddingsBuffers.constructor.name}), deEmbeddingsBuffers.length = ${deEmbeddingsBuffers.length}`);

//  console.log('embeddingsBuffers = ');
//  console.log(tq84_dumpObjectStructure(embeddingsBuffers));
//  console.log('deEmbeddingsBuffers = ');
//  console.log(tq84_dumpObjectStructure(deEmbeddingsBuffers));


    const { posEmbdBuffer } = await this.loadPositionalEmbeddings();
    console.log('posEmbdBuffer = ');
    console.log(tq84_dumpObjectStructure(posEmbdBuffer));

    this.layer_buffers  = await this.loadLayers();
    console.log('this.layer_buffers = ');
    console.log(tq84_dumpObjectStructure(this.layer_buffers));

//  console.log("Loading final layer norm...");
    const { normGammaBuffer, normBetaBuffer } = await this.loadFinalLayerNorm();
    console.log('normGammaBuffer = ');
    console.log(tq84_dumpObjectStructure(normGammaBuffer));
    console.log('normBetaBuffer = ');
    console.log(tq84_dumpObjectStructure(normBetaBuffer));

    const output = {posEmbdBuffer, normGammaBuffer, normBetaBuffer };
//  console.log('output = ');
//  console.log(tq84_dumpObjectStructure(output));

    console.log("Finished loading model.", output, this.params);
    return [output];
  }

  async loadParameters() {
     console.log('      loadParameters');

     this.params = await (await fetch(`${this.weightsFolder}/params_gpt.json`)).json();

     console.log(`       ${JSON.stringify(this.params)}`);
     console.log(`       vocab_size: ${this.params.vocab_size}`);
     console.log(`       n_embd:     ${this.params.n_embd}`);
     console.log(`       n_ctx:      ${this.params.n_ctx}`);

  // Did you enable GitHub LFS? Won't work without it.
     if (this.params.n_embd % 4 !== 0) throw new Error("Model load failed: n_embd must be divisible by 4.");
     if (this.params.n_embd % this.params.n_head !== 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

  // I'm unsure if this is a reasonable requirement here. At worst, I can figure out some padding method.
     if ((this.params.n_embd / this.params.n_head) % 4 !== 0) throw new Error("Model load failed: n_embd / n_head must be divisible by 4.");
     const tokenParam = this.bufferSize(this.params.vocab_size, this.params.n_embd);
     console.log(`       tokenParam = ${tokenParam} (vocab_size * n_embd * 4 = ${this.params.vocab_size * this.params.n_embd * 4})`);
     let minSplits = Math.ceil(tokenParam / this.device.limits.maxStorageBufferBindingSize);
     console.log(`       minSplits = ${minSplits}`);

     function vocabChunkSizeCalc(vocab_size, n_embd, splits, maxStorageBufferBindingSize) {
     // Possibly could be better? Needs actual benchmarking to know what approach is best.
       const optimisticSize = Math.ceil (vocab_size / splits / 4) * 4 * n_embd;
       const pessimiticSize = Math.floor(vocab_size / splits / 4) * 4 * n_embd;
       let vocab_chunk_size = optimisticSize;
       if (optimisticSize > maxStorageBufferBindingSize) {
         vocab_chunk_size = pessimiticSize;
         if (pessimiticSize * splits < tokenParam) {
           return vocabChunkSizeCalc(vocab_size, n_embd, splits + 1, maxStorageBufferBindingSize);
         }
       }
       return { vocab_chunk_size: vocab_chunk_size / n_embd, splits };
     }

     const { vocab_chunk_size, splits } = vocabChunkSizeCalc(this.params.vocab_size, this.params.n_embd, minSplits, this.device.limits.maxStorageBufferBindingSize);
     console.log(`       vocab_chunk_size = ${vocab_chunk_size}, splits = ${splits}`);

     if (splits > minSplits) console.warn(`Non-optimal number of vocab splits. Optimal: ${minSplits}, Selected: ${splits}`);

  // Set derived parameters
     this.params.vocab_chunk_size      = vocab_chunk_size;
     this.params.vocab_chunk_instances = splits;
     this.params.head_size             = this.params.n_embd / this.params.n_head;
     this.params.hidden_size           = this.params.n_embd * 4;
     this.params.attention_scale       = 1 / Math.sqrt(this.params.n_embd / this.params.n_head);
     this.params.bias                  = this.params.bias == undefined ? true : this.params.bias;

     // Check for overflow in buffers larger than maxStorageBufferBindingSize
     const maxBufferSize = this.device.limits.maxStorageBufferBindingSize / 4;
     if (this.params.n_embd * this.params.n_ctx                      > maxBufferSize) console.warn("Model load failed: n_embd * n_ctx must be less than maxStorageBufferBindingSize.");
     if (this.params.n_embd * this.params.hidden_size                > maxBufferSize) console.warn("Model load failed: n_embd * hidden_size must be less than maxStorageBufferBindingSize.");
     if (this.params.n_ctx  * this.params.n_ctx * this.params.n_head > maxBufferSize) console.warn("Model load failed: n_ctx * n_ctx must be less than maxStorageBufferBindingSize.");
     if (this.params.n_embd * this.params.n_embd * 3                 > maxBufferSize) console.warn("Model load failed: n_embd * n_embd * 3 must be less than maxStorageBufferBindingSize.");

     console.log("       Params:", this.params);

     return this.params;
  }

  async loadEmbeddings() {
    console.log('      loadEmbeddings');

    const embeddingWeights = await fetchBin2Float32Array(`${this.weightsFolder}/transformer.wte.weight_gpt.bin`);
    console.log(`       embeddingWeights length = ${embeddingWeights.length} byteLength = ${embeddingWeights.byteLength}`);

    // Chunks are stored in row-major order and are of dimensions n_embd x vocab_chunk_size.
    // Embedding weights are imported in column-major order and are of dimensions vocab_size x n_embd.
    // We pre-transpose the chunk for the deEmbedding process for the matmul. Could do this on GPU later.
    this.embeddingsBuffers   = [];
    this.deEmbeddingsBuffers = [];

    for (let i = 0; i < this.params.vocab_chunk_instances; i++) {
      console.log(`       Loading deEmbedding chunk ${i+1}/${this.params.vocab_chunk_instances}`);
      const offset = i * this.params.vocab_chunk_size;
      let size = this.params.vocab_chunk_size;
      console.log(`       offset = ${offset}, size = ${size}`);

      const paddedArray = new Float32Array(this.params.vocab_chunk_size * this.params.n_embd);
      if (i === this.params.vocab_chunk_instances - 1) {
        size = this.params.vocab_size - offset;
        console.log(`       i === this.params.vocab_chunk_instances -1: size = ${size} = this.params.vocab_size - offset`);
//      paddedArray.set(size * this.params.n_embd, zeros((this.params.vocab_chunk_size * this.params.vocab_chunk_instances - this.params.vocab_size) * this.params.n_embd));
//      paddedArray.set(zeros((this.params.vocab_chunk_size * this.params.vocab_chunk_instances - this.params.vocab_size) * this.params.n_embd), size * this.params.n_embd);
      }
      paddedArray.set(embeddingWeights.subarray(offset * this.params.n_embd, offset * this.params.n_embd + size * this.params.n_embd));

      this.embeddingsBuffers.push(this.initTensor(paddedArray, [this.params.vocab_chunk_size, this.params.n_embd], ["copy_from"]));

      const chunk = transpose(paddedArray, this.params.vocab_chunk_size, this.params.n_embd); // Use GPU perhaps?
      this.deEmbeddingsBuffers.push(this.initTensor(chunk, [this.params.n_embd, this.params.vocab_chunk_size], ["storage"]));
    }

//  return { embeddingsBuffers, deEmbeddingsBuffers };
  }

  async loadPositionalEmbeddings() {
//  console.log('      loadPositionalEmbeddings');
    const posEmbeddings = await fetchBin2Float32Array(`${this.weightsFolder}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, [this.params.n_ctx, this.params.n_embd], ["copy_from"]);

    return { posEmbdBuffer };
  }

  async loadFinalLayerNorm() {
//   console.log('      loadFinalyLayerNorm');
     const prefix = `${this.weightsFolder}/transformer.ln_f.`;

     const tensorPromises = [
        this.fetchAndInitTensor(`${prefix}weight_gpt.bin`, [this.params.n_embd], ["storage"]),
        this.fetchAndInitTensor(`${prefix}bias_gpt.bin`  , [this.params.n_embd], ["storage"]),
     ];

     const [normGammaBuffer, normBetaBuffer] = await Promise.all(tensorPromises);

     return { normGammaBuffer, normBetaBuffer };
  }

  async loadLayers() {
     console.log('      loadLayers');
     const layerPromises = [];

     for (let i = 0; i < this.params.n_layer; i++) {
       layerPromises.push(this.loadLayer(i));
     }

//   console.log('       *await Promise.all(layerPromises)')
     const layer_buffers = await Promise.all(layerPromises);
     console.log(`       *await ended, layer_buffers.length = ${layer_buffers.lengty}`)
     return layer_buffers;
  }

  async loadLayer(layerIndex) {
     const prefix = `${this.weightsFolder}transformer.h.${layerIndex}.`;

  // Create an array of promises for fetching and initializing the tensors
     const tensorPromises = [
        this.fetchAndInitTensor           (`${prefix}ln_1.weight_gpt.bin`       , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_1.bias_gpt.bin`         , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndSplitQKVWeightTensors(`${prefix}attn.c_attn.weight_gpt.bin`, [this.params.n_embd     , this.params.n_embd *3  ], ["storage"]),
        this.fetchAndSplitQKVBiasTensors  (`${prefix}attn.c_attn.bias_gpt.bin`  , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}attn.c_proj.weight_gpt.bin`, [this.params.n_embd     , this.params.n_embd     ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}attn.c_proj.bias_gpt.bin`  , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_2.weight_gpt.bin`       , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_2.bias_gpt.bin`         , [this.params.n_embd                              ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_fc.weight_gpt.bin`   , [this.params.n_embd     , this.params.hidden_size], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_fc.bias_gpt.bin`     , [this.params.hidden_size                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_proj.weight_gpt.bin` , [this.params.hidden_size, this.params.n_embd     ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_proj.bias_gpt.bin`   , [this.params.n_embd                              ], ["storage"]),
    ];

  // Wait for all tensors to be fetched and initialized
     const [
        normAttentionGammaBuffer,
        normAttentionBetaBuffer,
        qkvWeightArray,
        qkvBiasArray,
        linearWeightsBuffer,
        linearBiasBuffer,
        normLinearGammaBuffer,
        normLinearBetaBuffer,
        firstLayerWeightsBuffer,
        firstLayerBiasBuffer,
        secondLayerWeightsBuffer,
        secondLayerBiasBuffer,
     ] = await Promise.all(tensorPromises);

  // Process the fetched data and return the layer buffers
     return {
         normAttentionGammaBuffer,
         normAttentionBetaBuffer,
         qkvWeightArray,
         qkvBiasArray,
         linearWeightsBuffer,
         linearBiasBuffer,
         normLinearGammaBuffer,
         normLinearBetaBuffer,
         firstLayerWeightsBuffer,
         firstLayerBiasBuffer,
         secondLayerWeightsBuffer,
         secondLayerBiasBuffer,
     };
  }

  async fetchAndSplitQKVWeightTensors(url, dims, ops) {
    const data = transpose(await fetchBin2Float32Array(url), dims[0], dims[1]);

    const qWeights = transpose(data.subarray(0                    , dims[0] * dims[0]    ), dims[0], dims[0]);
    const kWeights = transpose(data.subarray(dims[0] * dims[0]    , dims[0] * dims[0] * 2), dims[0], dims[0]);
    const vWeights = transpose(data.subarray(dims[0] * dims[0] * 2, dims[0] * dims[0] * 3), dims[0], dims[0]);

    const qWeightsBuffer = this.initTensor(qWeights, [dims[0], dims[0]], ops);
    const kWeightsBuffer = this.initTensor(kWeights, [dims[0], dims[0]], ops);
    const vWeightsBuffer = this.initTensor(vWeights, [dims[0], dims[0]], ops);

    return [qWeightsBuffer, kWeightsBuffer, vWeightsBuffer];
  }

  async fetchAndSplitQKVBiasTensors(url, dims, ops) {
    const data = await fetchBin2Float32Array(url);

    const qBias = data.subarray(0, dims[0]);
    const kBias = data.subarray(dims[0], dims[0] * 2);
    const vBias = data.subarray(dims[0] * 2, dims[0] * 3);

    const qBiasBuffer = this.initTensor(qBias, [dims[0]], ops);
    const kBiasBuffer = this.initTensor(kBias, [dims[0]], ops);
    const vBiasBuffer = this.initTensor(vBias, [dims[0]], ops);

    return [qBiasBuffer, kBiasBuffer, vBiasBuffer];
  }

  async fetchAndInitTensor(url, dims, ops) {
//  console.log("Fetching and initializing tensor...", url);
    const data = await fetchBin2Float32Array(url);
    return this.initTensor(data, dims, ops);
  }

  initTensor(data, dims, ops) {
//
//  initTensor creates a GPUBuffer and returns it.
//
//  data: a  Float32Array
//  dims: an Array  
//  ops:  an Array
//

    let tq84_size = this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1);
    let tq84 = {
      size: tq84_size,
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    };

//  console.log(`          initTensor, data = ${data.constructor.name}, dims.length = ${dims.length}, ops.length = ${ops.length}`, tq84);
//  console.log(`          initTensor, data.length * 4 = ${data.length * 4}, size = ${tq84_size}`);

    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    const size = Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
//  console.log(`            buffersSize = ${Math.round(100*size/1024/1024)/100} MB = ${dimX} * ${dimY} * ${dimZ} * 4 (Float32Array.BYTES_PER_ELEMENT = ${Float32Array.BYTES_PER_ELEMENT}).`)
    if (size > this.device.limits.maxStorageBufferBindingSize)
//    console.warn("Warning: Buffer size calc result exceeds GPU limit, are you using this value for a tensor size?", dimX, dimY, dimZ, size);
      console.warn("                Warning: Buffer size exceeds GPU limit");
    return size;
  }
}
