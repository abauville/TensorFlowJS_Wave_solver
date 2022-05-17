# Wave solver using TensorFlow JS

# To do

- make an implicit formulation
- more colorful visu

# Resources

- [Tensor to image](https://www.oreilly.com/library/view/learning-tensorflowjs/9781492090786/ch04.html)


## Installation

TS.js [installation instructions](https://www.tensorflow.org/js/tutorials/setup).

Bundle with webpack (automatically rebuilds thanks to the `watch: true` config in `webpack.config.js`): 

```./node_modules/webpack/bin/webpack.js ./lib/index.js --output-filename=main.js --mode=development```

See [this tutorial](https://www.sitepoint.com/bundle-static-site-webpack/) to set up webpack.

Run a local server in another terminal: 
```python3 -m http.server```
