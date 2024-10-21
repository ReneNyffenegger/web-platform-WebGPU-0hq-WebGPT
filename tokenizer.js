class Tokenizer {
  constructor() {
    this.encoder = undefined;
    this.decoder = undefined;
    this.vocab_size = undefined;
  }

  async load() {
    throw new Error("Not implemented.");
  }

  getVocabSize() {
    return this.vocab_size;
  }

  encode(str) {
    throw new Error("Not implemented.");
  }

  decode(arr) {
    throw new Error("Not implemented.");
  }
}

class SimpleTokenizer extends Tokenizer {
  constructor() {
    super();
  }

  async load() {
    console.log("Loading simple tokenizer...");
    this.encoder = await (await fetch("weights/tokenization/simple_tokens.json")).json();
    this.decoder = Object.keys(this.encoder).reduce((acc, x) => ({ ...acc, [this.encoder[x]]: x }), {});
    this.vocab_size = Object.keys(this.encoder).length;
  }

  encode(str) {
    return str.split("").map((x) => this.encoder[x]);
  }

  decode(arr) {
    return arr.map((x) => this.decoder[x]).join("");
  }
}

// ------------------ GPT Tokenizer ------------------
// Credit to https://github.com/latitudegames/GPT-3-Encoder

class GPT2Tokenizer extends Tokenizer {
  constructor() {
    super();
    this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.textEncoder = new TextEncoder(); // always utf-8 by spec
    this.textDecoder = new TextDecoder("utf-8");
  }

  async load() {
    console.log('    Loading GPT2 tokenizer');
//  console.log("Loading GPT2 tokenizer...");

    const bpe_file = await (await fetch("weights/tokenization/vocab.bpe")).text();
    this.encoder = await (await fetch("weights/tokenization/gpt_tokens.json")).json();

 // The type of this.encoder (as well as also this.decoder) is Object.

    console.log('     Building decoder');
    this.decoder  = {};
    Object.keys(this.encoder).map((x) => {
      this.decoder[this.encoder[x]] = x;
    });

//  console.log(`    encoder = ${this.encoder.constructor.name}`, this.encoder);
//  console.log(`    decoder = ${this.decoder.constructor.name}`, this.decoder);

    const lines = bpe_file.split("\n");
    console.log(`     lines.length = ${lines.length}`);

 // Turn the lines in vocab.bpe into an array of arrays of two strings:
    const bpe_merges = lines.slice(1, lines.length - 1).map((x) => {
      return x.split(/\s+/); //.filter(function (e) {
    });

//  console.log(`     bpe_merges.length = ${bpe_merges.length}`);
//  console.log(`     bpe_merges[0] = ${bpe_merges[0]}`);
//  console.log(`     bpe_merges[1] = ${bpe_merges[1]}`);
//  console.log(`     bpe_merges[6] = ${bpe_merges[6]}`);

    const byte_encoder = bytes_to_unicode();
    const byte_decoder = {};
    Object.keys(byte_encoder).map((x) => {
      byte_decoder[byte_encoder[x]] = x;
    });
    this.byte_encoder = byte_encoder;
    this.byte_decoder = byte_decoder;

    this.bpe_ranks = dictZip(bpe_merges, range(0, bpe_merges.length));

    this.cache = new Map();
    this.vocab_size = Object.keys(this.encoder).length;
  }

  encode(text) {
    console.log(`GPT2Tokenizer.encode ${text}`);
    if (!this.byte_encoder) throw new Error("Tokenizer not loaded.");
    let bpe_tokens = [];
    const matches = Array.from(text.matchAll(this.pat)).map((x) => x[0]);
    for (let match of matches) {
      console.log(` match = >${match}<`);

    // textEncoder.encode() takes a string and returns
    // a Uint8Array with the text encoded as UTF8:
      const utf8_bytes = this.textEncoder.encode(match);

      let utf8_bytes_printable = [];
      for (let i = 0; i < utf8_bytes.length; i++) {
//      console.log(` encoded_byte = ${utf8_bytes[i]} pushed: ${this.byte_encoder[utf8_bytes[i]]}`);
        utf8_bytes_printable.push(this.byte_encoder[utf8_bytes[i]]);
      }
      let bytes_str = utf8_bytes_printable.join("");

      const new_tokens = this.bpe(bytes_str)
        .map((x) => this.encoder[x]);

      bpe_tokens = bpe_tokens.concat(new_tokens);
    }
    return bpe_tokens;
  }

  decode(tokens) {
    if (!this.byte_decoder) throw new Error("Tokenizer not loaded.");
    let text = tokens.map((x) => this.decoder[x]).join("");
    text = this.textDecoder.decode(new Uint8Array(text.split("").map((x) => this.byte_decoder[x])));
    return text;
  }

  bpe(token) {
 // token is a string and an array of strings is returned.
    console.log(`  bpe, token = >${token}< (${token.constructor.name})`);
    if (this.cache.has(token)) {
       console.log(`   token is cached, returning ${this.cache.get(token)} type = ${this.cache.get(token).constructor.name}`);
       return this.cache.get(token);
    }

    let word = token.split("");
    console.log('   word = ', word);

    let pairs = get_bigrams(word);

    console.log('   pairs = ', pairs);
    if (!pairs) {
       alert('is pairs ever !pairs?'); // Is this alert ever shown?
       console.log('   !pairs, return token');
       return token;
    }

    while (true) {
      console.log('   -- next iteration')

   // While we're iterating over each pair,
   // we store the pair's rank as key and the pair
   // itself as value in this dictionary:
      const rank2pair = {};

      pairs.forEach(pair => {
//      console.log('   forEach(pair) = ', pair);
        const rank = this.bpe_ranks[pair];
//      console.log('   rank = ', rank);
        rank2pair[isNaN(rank) ? 10e10 : rank] = pair;
      });


   // Even though the dictionary's elements were assigned with
   // integer keys, they seem to be stored with strings.
      const ranks = Object.keys(rank2pair).map((x) => parseInt(x));
//    console.log('   ranks = ', ranks);


   // Find the pair (bigram) with the minimal rank:
      const bigram_min_rank = rank2pair[Math.min(...ranks)];
      console.log('   bigram_min_rank = ', bigram_min_rank);
      if (!Object.hasOwn(this.bpe_ranks, bigram_min_rank)) {
   //
   // The bigram with the minimal rank is not found in vocab.bpe,
   // we're brekaing out of the loop.
   // TODO, is this not the same as ranks being equal to [10e10] ?
//       console.log(`   !Object.hasOwn ${bigram_min_rank}, breaking out`);
         break;
      }


      const first  = bigram_min_rank[0];
      const second = bigram_min_rank[1];
//    console.log(`   first = ${first}, second = ${second}`);

      let new_word = [];
      let i = 0;
      while (i < word.length) {
        console.log(`   word[${i}] = ${word[i]}`);
        const j = word.indexOf(first, i);
      
        if (j === -1) {
          new_word = new_word.concat(word.slice(i));
          break;
        }

        new_word = new_word.concat(word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          new_word.push(first + second);
          i = i + 2;
        }
        else {
          new_word.push(word[i]);
          i = i + 1;
        }
      }
      word = new_word;
      if (word.length === 1) {
         console.log('   Word.length === 1, breaking out');
         break;
      }
      else {
         pairs = get_bigrams(word);
      }
    }

    const bpe_ret = word;
    this.cache.set(token, bpe_ret);
    console.log(`   bpe, returning ${bpe_ret} (${bpe_ret.constructor.name})`);
    return bpe_ret;
  }
}

const range = (x, y) => {
  const res = [];
  for (let i = x; i < y; i++) { res.push(i) }
  return res;
};

const ord = (x) => {
  return x.charCodeAt(0);
};

const dictZip = (x, y) => {
//
//    Create a dictionary from two arrays of which
//    the first specifies the keys and the second
//    the values:
//
//      dictZip(['a', 'b', 'c'], ['x', 'y', 'z']) --> {a: 'x', b: 'y', c: 'z'}
//
  const result = {};
  x.map((_, i) => {
    result[x[i]] = y[i];
  });
  return result;
};

const bytes_to_unicode = () => {
   console.log(`  bytes_to_unicode`);
   const bs = range(ord("!"), ord("~") + 1).concat(range(ord("¡"), ord("¬") + 1), range(ord("®"), ord("ÿ") + 1));

//
// Copy bs
   let cs = bs.slice();


//
// Initially, both, bs and cs are arrays with 188 elements.
//
// console.log(`   bs.length = ${bs.length} cs.length = ${cs.length}, bs.constructor.name = ${bs.constructor.name}, cs.constructor.name = ${cs.constructor.name}`, bs, cs);

   let n = 0;
   for (let b = 0; b < 2 ** 8; b++) {  // for b = 0 .. 255
// console.log(`   b=${b}`);
      if (!bs.includes(b)) {
//       console.log(`   b is not included in bs`);
         bs.push(b);
         cs.push(2 ** 8 + n);
         n = n + 1;
      }
   }

//
// Now, their length has increased to 256:
//
// console.log(`   bs.length = ${bs.length} cs.length = ${cs.length}, bs.constructor.name = ${bs.constructor.name}, cs.constructor.name = ${cs.constructor.name}`, bs, cs);

   cs = cs.map((x) => String.fromCharCode(x));
   const result = {};
   bs.map((_, i) => {
      result[bs[i]] = cs[i];
   });

//
// result is an Object whose keys are the integers 0 .. 255
// Each value is 'printable Unicode character':
//    {
//       0: 'Ā',
//       1: 'ā',
//       2: 'Ă',
//       3: 'ă',
//       4: 'Ą',
//
//          ....
//
//      32: 'Ġ',
//      33: '!',
//      34: '"',
//      35: '#'
//
//          ...
//    }
//
// console.log(`   bytes_to_unicode: length of result = ${result.length}, constructor name = ${result.constructor.name}`, result);
   return result;
};

const get_bigrams = (grams) => {
//
//    grams is an array of 'printable characters' or already concatenated printable characters
//    get_bigrams returns a set in which each element is an Array(2).
//    Each array stores the neighboring printable characters.
//
//    TODO: I've renamed get_words to get_bigrams.
//          Is the change of the name correct?
//
  console.log(`    get_bigrams, grams = `, grams);
  const pairs = new Set();
  let prev_char = grams[0];
  for (let i = 1; i < grams.length; i++) {
    const char = grams[i];
    pairs.add([prev_char, char]);
    prev_char = char;
  }
  console.log(`     returning pairs = `, pairs);
  return pairs;
};
