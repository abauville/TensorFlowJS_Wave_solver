import * as tf from '@tensorflow/tfjs';
import { loadGraphModel, StringToHashBucketFast } from '@tensorflow/tfjs';

export function getIniTensor(nx=20, ny=20, sigma=0.2) {
  // let a = tf.zeros([1,1]);
  return tf.tidy(() => { 
    const x = tf.linspace(-1.0, 1.0, nx);
    const y = tf.linspace(-1.0, 1.0, ny);
    let X;
    let Y;
    [X, Y] = tf.meshgrid(x, y);
    // 2D gaussian
    return ( 
      X.square().mul(-0.5/(sigma*sigma)).exp().mul(
        Y.square().mul(-0.5/(sigma*sigma)).exp()
      )
    ).reshape([nx,ny,1]);
  });
}

export function delOperator(scale = 0.249) {
  return tf.tidy(() => {
    return (
      tf.tensor([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
      ])
      .reshape([3,3,1,1])
      .mul(scale)
    );
  });
}


export function del(A, scale=0.249) {
  const filter = delOperator(scale);
  return tf.tidy(() => {
    return tf.pad(
      tf.conv2d(A,filter,1,'valid'), 
      [[1,1], [1,1], [0,0]]
    );
  })
}