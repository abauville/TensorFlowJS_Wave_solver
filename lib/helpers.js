import * as tf from '@tensorflow/tfjs';
import { loadGraphModel, StringToHashBucketFast } from '@tensorflow/tfjs';

export function getIniTensor(nx=20, ny=20, sigma=0.2) {
  const x = tf.linspace(-1.0, 1.0, nx);
  const y = tf.linspace(-1.0, 1.0, ny);
  let X;
  let Y;
  [X, Y] = tf.meshgrid(x, y);
  let a = tf.tidy(() => { 
    // 2D gaussian
    return ( 
      X.square().mul(-0.5/(sigma*sigma)).exp().mul(
        Y.square().mul(-0.5/(sigma*sigma)).exp()
      )
    ).reshape([nx,ny,1]);
  });
  // a.print();
  x.dispose();
  y.dispose();
  X.dispose();
  Y.dispose();
  return a
}

export function getFilter(scale = 0.249) {
  let filter = tf.tensor([
    [0, 1, 0],
    [1,-4, 1],
    [0, 1, 0]
  ]).reshape([3,3,1,1])
  return filter.mul(scale);
}



export function runStep(A, filter) {
  const out = tf.conv2d(A,filter,1,'valid');
  const out_padded = tf.pad(out,[[1,1], [1,1], [0,0]]);
  A = A.add(out_padded)
  out.dispose();
  out_padded.dispose();
  return A;
}