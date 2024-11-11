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

// ---------------------------------------------------

function tq84_assert(shouldBeTrue, textIfNotTrue) {
   if (! shouldBeTrue) throw new Error(textIfNotTrue);
}
function tq84_assertClass(obj, className) {
   tq84_assert(obj.constructor.name == className, `expected className = ${className}, but got ${obj.constructor.name}`);
}

// ---------------------------------------------------


function stack_depth() {
   const e = new Error('stack trace');
// alert(e.stack);
   const nofNewLines = (e.stack.match(/\n/g) || []).length;
// alert(nofNewLines);
   return nofNewLines - 2;
}

class tq84_log {

   async init(suggestedName, id) {
      const fileh    = await window.showSaveFilePicker( {id: id, suggestedName: suggestedName});
      this.outstream = await fileh.createWritable();
//    this.x         = await fileh.createSyncAccessHandle();
   }

   async enter(txt) {
      await this.out_(txt, stack_depth() * 2);
//    console.log('e.stack', e.stack.constructor.name); // String
//    console.log('e.stack.length', e.stack.length);

//    console.log('e.stack', e.stack);
//    await this.outstream.write('  '.repeat(nofNewLines) + txt + "\n");

//    alert('  '.repeat(e.stack.length) + txt);
   }

   async log(txt) {
      await this.out_(txt, stack_depth() * 2 + 1);
//    console.log('e.stack', e.stack.constructor.name); // String
//    console.log('e.stack.length', e.stack.length);

//    console.log('e.stack', e.stack);
//    await this.outstream.write('  '.repeat(nofNewLines) + txt + "\n");

//    alert('  '.repeat(e.stack.length) + txt);
   }


   async end() {
      this.outstream.close();
   }


   async out_(txt, indent) {
      await this.outstream.write(' '.repeat(indent) + txt + "\n");
   }



}
