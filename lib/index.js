import * as tf from '@tensorflow/tfjs';
import * as p5 from 'p5';
import { loadGraphModel, StringToHashBucketFast } from '@tensorflow/tfjs';
import {getIniTensor, del} from "/lib/helpers.js";


const p_fps = document.querySelector('#fps');
let counter = 0
const fps_array = new Array(30).fill(0);

function mean(array) {
  const sum = array.reduce((a, b) => a + b, 0);
  return (sum / array.length) || 0;
}

const sketch = (p) => {
  p.setup = () => {
    const canvas = p.createCanvas(WIDTH, HEIGHT, 'WEBGL');
    const container = document.querySelector(".main-container");
    container.style.width = `${WIDTH}px`;
    container.style.height = `${HEIGHT}px`;
    canvas.parent('sketch-container');
  }

  p.mousePressed = () => {
    p.redraw();
  }
  p.draw = () => {
    fps_array[counter%fps_array.length] = 1000/p.deltaTime
    p_fps.innerText = `${Math.round(mean(fps_array))}`;
    counter += 1;
    // for (let i=0; i<nsub; i++) {
    //   dHdt = tf.tidy( () => dHdt.add(del(H)));
    //   H = H.add(dHdt);
    // }
    
    Hplot = tf.tidy(() => H.mul(2.0).add(0.5).clipByValue(0,1));
    tf.browser.toPixels(Hplot, p.canvas).then(() => {   
      
    });
    if (counter > nt) {
      p.noLoop();
      H.dispose();
      Hplot.dispose();
      dHdt.dispose();
      console.log("memory.numTensors:", tf.memory().numTensors);
    }
  }
}


// Run
// ========
const WIDTH = 600
const HEIGHT = 600
const nt = 1; // max number of steps
const nsub = 15; // number of substeps (i.e. computation steps between within each draw iteration)
let dHdt = tf.zeros([WIDTH, HEIGHT, 1]);
let H = getIniTensor(WIDTH,HEIGHT,0.01);
let Hplot;
const dt = 0.99;
console.log("memory.numTensors:", tf.memory().numTensors);
const sketchP = new p5(sketch);
