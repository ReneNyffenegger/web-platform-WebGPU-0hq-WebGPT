const webgpuSupportMessage = document.getElementById("webgpuSupportMessage");
const loadModelButton      = document.getElementsByClassName('loadModelButton');

// alert(loadModelButton.constructor.name); // HTMLCollection

const setModelButtonDisabled = (bool) => {
   for (let e of loadModelButton) {
      e.disabled = bool;
   }
// for (let i = 0; i < loadModelButton.length; i++) loadModelButton[i].disabled = bool;
};

const destroyCacheButton = document.getElementById("destroyCacheButton");

const nofTokensToGenerate = document.getElementById("nofTokensToGenerate");
const topKInput = document.getElementById("topKInput");
const temperatureInput = document.getElementById("temperatureInput");
const generateButton = document.getElementById("generateButton");
const promptTextarea = document.getElementById("prompt");
const updateVisualsButton = document.getElementById("updateVisuals");

let GPTModel = null;
let visuals = null;

// Check for WebGPU support
if (!navigator.gpu) {
   webgpuSupportMessage.textContent = 'WebGPU is not supported in your browser.'; //  Update Chrome to v113 or download <a href='https://www.google.com/chrome/canary/'>Chrome Canary</a>. Also available on <a href='https://www.microsoftedgeinsider.com/en-us/download/canary'>Edge Canary</a>.";
}
else {
   webgpuSupportMessage.textContent = "WebGPU is supported in your browser!";
   setModelButtonDisabled(false);
//      tq84_init();
}

async function startGeneration() {
   setTextareaDisabled(true);

   tq84_log_ = new tq84_log();
   await tq84_log_.init('gpu-log.hlg', 'gpu-log');
   await tq84_log_.enter('startGeneration');

   const prompt      = promptTextarea.value || " ";
   const numTokens   = nofTokensToGenerate.value;
   const topK        = topKInput.value;
   const temperature = temperatureInput.value;
   await tq84_log_.log(`numTokens = {numTokens}, topK = {topK}`);
   const textStream  = GPTModel.generate(prompt, numTokens, topK, temperature);

   for await (const text of textStream) {
      promptTextarea.value += text;
      visuals.render();
   }

   setTextareaDisabled(false);
   await tq84_log_.end();
}

async function loadModel(folder, tokenizer) {
   console.log(`loadModel, folder = ${folder}`);
   setModelButtonDisabled(true);

   console.log(` new GPT`);
   GPTModel = new GPT(folder, tokenizer);
   await GPTModel.initialize();
   console.log(` GPT model is initialized`);


   promptTextarea.value = GPTModel.defaultPrompt;
   topKInput.value = GPTModel.defaultTopK;
   nofTokensToGenerate.value = GPTModel.defaultTokens;
   temperatureInput.value = GPTModel.defaultTemperature;

   setTextareaDisabled(false);
   nofTokensToGenerate.disabled = false;
   topKInput.disabled = false;
   temperatureInput.disabled = false;

   destroyCacheButton.disabled = false;
   updateVisualsButton.disabled = false;

   if (!visuals) {
     console.log(` !visuals, create one`);
     visuals = new Visuals(GPTModel);
     visuals.init();
     visuals.render();
   }
}

function setTextareaDisabled(bool) {
  promptTextarea.disabled = bool;
  generateButton.disabled = bool;
}

async function continueGeneration() {
  setTextareaDisabled(true);

  const prompt = outputDiv.innerHTML || " ";
  const numTokens = nofTokensToGenerate.value;

  outputDiv.innerHTML = prompt;

  const topK = topKInput.value;
  const temperature = temperatureInput.value;
  const textStream = generate(prompt, numTokens, 10, topK, temperature);

  for await (const text of textStream) {
    outputDiv.innerHTML += text;
  }

  setTextareaDisabled(false);
}

function destroyCache() {
  GPTModel.unloadBuffers();
  destroyOperations();

  GPTModel = null;

  visuals.destroy();
  visuals = null;

  setModelButtonDisabled(false);

  destroyCacheButton.disabled = true;
  nofTokensToGenerate.disabled = true;
  topKInput.disabled = true;
  temperatureInput.disabled = true;
  updateVisualsButton.disabled = true;
  setTextareaDisabled(true);
}

function updateVisuals() {
    visuals.render();
}

async function tq84_init() {
   tk84 = new GPT2Tokenizer();
   await tk84.load();

   tk84.encode('1234abcd änd  1234 abcd jeiodlqbkdalkf 1234');
}
