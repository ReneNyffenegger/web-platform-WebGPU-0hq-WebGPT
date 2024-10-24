
function tq84_spaces(n) { return ' '.repeat(n*2); }
function tq84_dumpObjectStructure(obj, indent=0) {
// console.log(tq84_dumpObjectStructure( [1, 2, 3 ]))
// console.log(tq84_dumpObjectStructure( [1, ['a', 'b', 'c'], 3 ]))
// console.log(tq84_dumpObjectStructure( [1, 2, ['level', 'one', 'array'], {x: null, y: "foo bar baz", z:{ ary: ['x', 'y', navigator.gpu, 'z'], emptyObj: {} }} ]  ))

   if (Array.isArray(obj)) {
      return "[\n" +
         obj.map( item => tq84_spaces(indent+1) + tq84_dumpObjectStructure(item, indent+1)).join(",\n") +
        "\n" + tq84_spaces(indent) + "]";
   }

   if (typeof obj === 'object' && obj !== null /* && obj.constructor.name == 'Object' */ ) {

      return "{\n" +
        Object.keys(obj).map(
           k => tq84_spaces(indent+1) + k + ': ' + tq84_dumpObjectStructure(obj[k], indent+1)
        ).join(",\n" ) +
      "\n" + tq84_spaces(indent) + "}";

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

    [this.model, this.params] = await this.loadModel(this.folder);
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
      this.defaultTokens = 80;
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
//    const { passes, resultBuffer } = EmbedBlock.newInstance(idx, idx.length, this.params.n_embd, this.params.vocab_chunk_size, this.model.embeddingsBuffers, this.model.posEmbdBuffer, ResidualBlock);
      const { passes, resultBuffer } = EmbedBlock.newInstance(idx,             this.params.n_embd, this.params.vocab_chunk_size, this.model.embeddingsBuffers, this.model.posEmbdBuffer, ResidualBlock);

      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      this.computePasses.push(...passes);
    }


    for (let layer = 0; layer < this.params.n_layer; layer++) {

      const buffers = this.model.layer_buffers[layer];

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
        this.model.deEmbeddingsBuffers
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
    const weightsFolder = `weights/${folder}/`;

    const params = await this.loadParameters(weightsFolder);
    const { embeddingsBuffers, deEmbeddingsBuffers } = await this.loadEmbeddings(params, weightsFolder);
    console.log(`     loadEmbeddings returned embeddingsBuffers.length = ${embeddingsBuffers.length} (${embeddingsBuffers.constructor.name}), deEmbeddingsBuffers.length = ${deEmbeddingsBuffers.length}`);

    console.log('embeddingsBuffers = ');
    console.log(tq84_dumpObjectStructure(embeddingsBuffers));
    console.log('deEmbeddingsBuffers = ');
    console.log(tq84_dumpObjectStructure(deEmbeddingsBuffers));


    const { posEmbdBuffer } = await this.loadPositionalEmbeddings(params, weightsFolder);
    const layer_buffers = await this.loadLayers(params, weightsFolder);

//  console.log("Loading final layer norm...");
    const { normGammaBuffer, normBetaBuffer } = await this.loadFinalLayerNorm(params, weightsFolder);

    const output = { layer_buffers, embeddingsBuffers, deEmbeddingsBuffers, posEmbdBuffer, normGammaBuffer, normBetaBuffer };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async loadParameters(weightsFolder) {
     console.log('      loadParameters');
//   console.log("Loading params...");
     const params = await (await fetch(`${weightsFolder}/params_gpt.json`)).json();
     console.log(`       ${JSON.stringify(params)}`);
     console.log(`       vocab_size: ${params.vocab_size}`);
     console.log(`       n_embd:     ${params.n_embd}`);
     console.log(`       n_ctx:      ${params.n_ctx}`);

  // Did you enable GitHub LFS? Won't work without it.
     if (params.n_embd % 4 !== 0) throw new Error("Model load failed: n_embd must be divisible by 4.");
     if (params.n_embd % params.n_head !== 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");

  // I'm unsure if this is a reasonable requirement here. At worst, I can figure out some padding method.
     if ((params.n_embd / params.n_head) % 4 !== 0) throw new Error("Model load failed: n_embd / n_head must be divisible by 4.");
     const tokenParam = this.bufferSize(params.vocab_size, params.n_embd);
     console.log(`       tokenParam = ${tokenParam} (vocab_size * n_embd * 4 = ${params.vocab_size * params.n_embd * 4})`);
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

     const { vocab_chunk_size, splits } = vocabChunkSizeCalc(params.vocab_size, params.n_embd, minSplits, this.device.limits.maxStorageBufferBindingSize);
     console.log(`       vocab_chunk_size = ${vocab_chunk_size}, splits = ${splits}`);

     if (splits > minSplits) console.warn(`Non-optimal number of vocab splits. Optimal: ${minSplits}, Selected: ${splits}`);

  // Set derived parameters
     params.vocab_chunk_size      = vocab_chunk_size;
     params.vocab_chunk_instances = splits;
     params.head_size             = params.n_embd / params.n_head;
     params.hidden_size           = params.n_embd * 4;
     params.attention_scale       = 1 / Math.sqrt(params.n_embd / params.n_head);
     params.bias                  = params.bias == undefined ? true : params.bias;

     // Check for overflow in buffers larger than maxStorageBufferBindingSize
     const maxBufferSize = this.device.limits.maxStorageBufferBindingSize / 4;
     if (params.n_embd * params.n_ctx                 > maxBufferSize) console.warn("Model load failed: n_embd * n_ctx must be less than maxStorageBufferBindingSize.");
     if (params.n_embd * params.hidden_size           > maxBufferSize) console.warn("Model load failed: n_embd * hidden_size must be less than maxStorageBufferBindingSize.");
     if (params.n_ctx  * params.n_ctx * params.n_head > maxBufferSize) console.warn("Model load failed: n_ctx * n_ctx must be less than maxStorageBufferBindingSize.");
     if (params.n_embd * params.n_embd * 3            > maxBufferSize) console.warn("Model load failed: n_embd * n_embd * 3 must be less than maxStorageBufferBindingSize.");

//   console.log("Params:", params);
     console.log("       Params:", params);

     return params;
  }

  async loadEmbeddings(params, weightsFolder) {
    console.log('      loadEmbeddings');
//  console.log("Loading token embeddings...");
    const embeddingWeights = await fetchBin2Float32Array(`${weightsFolder}/transformer.wte.weight_gpt.bin`);
    console.log(`       embeddingWeights length = ${embeddingWeights.length} byteLength = ${embeddingWeights.byteLength}`);

    // Chunks are stored in row-major order and are of dimensions n_embd x vocab_chunk_size.
    // Embedding weights are imported in column-major order and are of dimensions vocab_size x n_embd.
    // We pre-transpose the chunk for the deEmbedding process for the matmul. Could do this on GPU later.
    const embeddingsBuffers = [];
    const deEmbeddingsBuffers = [];
    for (let i = 0; i < params.vocab_chunk_instances; i++) {
      console.log(`       Loading deEmbedding chunk ${i+1}/${params.vocab_chunk_instances}`);
//    console.log(`Loading deEmbedding chunk ${i + 1}/${params.vocab_chunk_instances}...`);
      const offset = i * params.vocab_chunk_size;
      let size = params.vocab_chunk_size;
      console.log(`       offset = ${offset}, size = ${size}`);

      const paddedArray = new Float32Array(params.vocab_chunk_size * params.n_embd);
      if (i === params.vocab_chunk_instances - 1) {
        size = params.vocab_size - offset;
        console.log(`       i === params.vocab_chunk_instances -1: size = ${size} = params.vocab_size - offset`);
//      paddedArray.set(size * params.n_embd, zeros((params.vocab_chunk_size * params.vocab_chunk_instances - params.vocab_size) * params.n_embd));
//      paddedArray.set(zeros((params.vocab_chunk_size * params.vocab_chunk_instances - params.vocab_size) * params.n_embd), size * params.n_embd);
      }
      paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));

      embeddingsBuffers.push(this.initTensor(paddedArray, [params.vocab_chunk_size, params.n_embd], ["copy_from"]));

      const chunk = transpose(paddedArray, params.vocab_chunk_size, params.n_embd); // Use GPU perhaps?
      deEmbeddingsBuffers.push(this.initTensor(chunk, [params.n_embd, params.vocab_chunk_size], ["storage"]));
    }

    return { embeddingsBuffers, deEmbeddingsBuffers };
  }

  async loadPositionalEmbeddings(params, weightsFolder) {
//  console.log("Loading positional embeddings...");
//  console.log('      loadPositionalEmbeddings');
    const posEmbeddings = await fetchBin2Float32Array(`${weightsFolder}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, [params.n_ctx, params.n_embd], ["copy_from"]);

    return { posEmbdBuffer };
  }

  async loadFinalLayerNorm(params, weightsFolder) {
//   console.log("Loading final norm...");
//   console.log('      loadFinalyLayerNorm');
     const prefix = `${weightsFolder}/transformer.ln_f.`;

     const tensorPromises = [
        this.fetchAndInitTensor(`${prefix}weight_gpt.bin`, [params.n_embd], ["storage"]),
        this.fetchAndInitTensor(`${prefix}bias_gpt.bin`, [params.n_embd], ["storage"]),
     ];

     const [normGammaBuffer, normBetaBuffer] = await Promise.all(tensorPromises);

     return { normGammaBuffer, normBetaBuffer };
  }

  async loadLayers(params, weightsFolder) {
//   console.log("Loading layers...");
     console.log('      loadLayers');
     const layerPromises = [];

     for (let i = 0; i < params.n_layer; i++) {
       layerPromises.push(this.loadLayer(params, weightsFolder, i));
     }

//   console.log('       *await Promise.all(layerPromises)')
     const layer_buffers = await Promise.all(layerPromises);
     console.log(`       *await ended, layer_buffers.length = ${layer_buffers.lengty}`)
     return layer_buffers;
  }

  async loadLayer(params, weightsFolder, layerIndex) {
//   console.log("Starting to load layer...", layerIndex);
     const prefix = `${weightsFolder}transformer.h.${layerIndex}.`;

  // Create an array of promises for fetching and initializing the tensors
     const tensorPromises = [
        this.fetchAndInitTensor           (`${prefix}ln_1.weight_gpt.bin`       , [params.n_embd                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_1.bias_gpt.bin`         , [params.n_embd                         ], ["storage"]),
        this.fetchAndSplitQKVWeightTensors(`${prefix}attn.c_attn.weight_gpt.bin`, [params.n_embd     , params.n_embd *3  ], ["storage"]),
        this.fetchAndSplitQKVBiasTensors  (`${prefix}attn.c_attn.bias_gpt.bin`  , [params.n_embd                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}attn.c_proj.weight_gpt.bin`, [params.n_embd     , params.n_embd     ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}attn.c_proj.bias_gpt.bin`  , [params.n_embd                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_2.weight_gpt.bin`       , [params.n_embd                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}ln_2.bias_gpt.bin`         , [params.n_embd                         ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_fc.weight_gpt.bin`   , [params.n_embd     , params.hidden_size], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_fc.bias_gpt.bin`     , [params.hidden_size                    ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_proj.weight_gpt.bin` , [params.hidden_size, params.n_embd     ], ["storage"]),
        this.fetchAndInitTensor           (`${prefix}mlp.c_proj.bias_gpt.bin`   , [params.n_embd                         ], ["storage"]),
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
