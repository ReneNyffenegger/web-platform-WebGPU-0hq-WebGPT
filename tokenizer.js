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
    const encoder = await (await fetch("weights/tokenization/gpt_tokens.json")).json();
    this.encoder = encoder;

//  console.log("Building decoder...");
    console.log('     Building decoder');
    const decoder = {};
    Object.keys(encoder).map((x) => {
      decoder[encoder[x]] = x;
    });
    this.decoder = decoder;

    const lines = bpe_file.split("\n");
    console.log(`     lines.length = ${lines.length}`);
    const bpe_merges = lines.slice(1, lines.length - 1).map((x) => {
      return x.split(/(\s+)/).filter(function (e) {
        return e.trim().length > 0;
      });
    });
    console.log(`     bpe_merges.length = ${bpe_merges.length}`);

    const byte_encoder = bytes_to_unicode();
    const byte_decoder = {};
    Object.keys(byte_encoder).map((x) => {
      byte_decoder[byte_encoder[x]] = x;
    });
    this.byte_encoder = byte_encoder;
    this.byte_decoder = byte_decoder;

    this.bpe_ranks = dictZip(bpe_merges, range(0, bpe_merges.length));
    this.cache = new Map();
    this.vocab_size = Object.keys(encoder).length;
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
        console.log(` encoded_byte = ${utf8_bytes[i]} pushed: ${this.byte_encoder[utf8_bytes[i]]}`);
//      utf8_bytes_printable.push(this.byte_encoder[utf8_bytes[i].toString()]);
        utf8_bytes_printable.push(this.byte_encoder[utf8_bytes[i]           ]);
      }
      let bytes_str = utf8_bytes_printable.join("");

      const new_tokens = this.bpe(bytes_str)
        .split(" ")
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
  // token is a string and a string is returned.
    console.log(`  bpe, token = >${token}< (${token.constructor.name})`);
    if (this.cache.has(token)) {
       console.log(`   token is cached, returning ${this.cache.get(token)} type = ${this.cache.get(token).constructor.name}`);
       return this.cache.get(token);
    }
    let word = token.split("");
    console.log(`   word = `, word);
    let pairs = get_pairs(word);
    if (!pairs) return token;
    while (true) {
      const minPairs = {};
      pairs.forEach(pair => {
        const rank = this.bpe_ranks[pair];
        minPairs[isNaN(rank) ? 10e10 : rank] = pair;
      });
      const keys = Object.keys(minPairs).map((x) => parseInt(x));
      const bigram = minPairs[Math.min(...keys)];
      if (!Object.hasOwn(this.bpe_ranks, bigram)) break;
      const first = bigram[0];
      const second = bigram[1];
      let new_word = [];
      let i = 0;
      while (i < word.length) {
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
        } else {
          new_word.push(word[i]);
          i = i + 1;
        }
      }
      word = new_word;
      if (word.length === 1) break;
      else pairs = get_pairs(word);
    }
    word = word.join(" ");
    this.cache.set(token, word);
    return word;
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
//                      33 ,     126                        161 ,     172                  174 ,    255
  let cs = bs.slice();
  console.log(`   bs.length = ${bs.length} cs.length = ${cs.length}`);
  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
    console.log(`   b=${b}`);
    if (!bs.includes(b)) {
      console.log(`   b is not included in bs`);
      bs.push(b);
      cs.push(2 ** 8 + n);
      n = n + 1;
    }
  }
  cs = cs.map((x) => String.fromCharCode(x));
  const result = {};
  bs.map((_, i) => {
    result[bs[i]] = cs[i];
  });
  return result;
};

const get_pairs = (word) => {
//
//    word is an array of 'printable characters'
//    get_pairs returns a set in which each element is an Array(2).
//    Each array stores the neighboring printable characters.
//
  console.log(`    get_pairs, word = `, word);
  const pairs = new Set();
  let prev_char = word[0];
  for (let i = 1; i < word.length; i++) {
    const char = word[i];
    pairs.add([prev_char, char]);
    prev_char = char;
  }
  console.log(`     returning pairs = `, pairs);
  return pairs;
};
