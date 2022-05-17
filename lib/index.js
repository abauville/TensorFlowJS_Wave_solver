import * as tf from '@tensorflow/tfjs';
import * as p5 from 'p5';
import { loadGraphModel, StringToHashBucketFast } from '@tensorflow/tfjs';
import {getIniTensor, getFilter, runStep} from "/lib/helpers.js";


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
    for (let i=0; i<nsub; i++) {
      T = runStep(T, filter);
    }
    
    
    tf.browser.toPixels(T, p.canvas).then(() => {   
    });
    if (counter > nt) {
      p.noLoop();
      T.dispose();
    }
  }
}


// Run
// ========
const WIDTH = 600
const HEIGHT = 600
const nt = 100 // max number of steps
const nsub = 10; // number of substeps (i.e. computation steps between within each draw iteration)
let T = getIniTensor(WIDTH,HEIGHT,0.1);
const filter = getFilter(0.2499);
const sketchP = new p5(sketch);
