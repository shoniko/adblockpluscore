/** @module ml */

"use strict";


const tf = require("@tensorflow/tfjs");

/**
 * Class for Machine Learning functions
 * @constructor
 */
function MachineLearning()
{
  browser.webNavigation.onCompleted.addListener(details =>
  {
    if (this.curNavigation > this.trainingThreshold)
    {
      // We retrain our ML model on every `trainingThreshold` navigations
      exports.MlModel.fit();
      // Reset the counter
      this.curNavigation = 0;
      return;
    }
    this.curNavigation ++;
  });

  this.buildModel();
}

MachineLearning.prototype = {

  data: [],
  labels: [],
  model: null,

  // Used to store averaged weights
  federatedWeights: [],

  // Used to store weights of the last training
  intermediateWeights: [],

  // Used to keep track of how many training have we done
  curIteration: 0,

  // Used to keep track of how many navigations have happened
  // since the last training
  curNavigation: 0,

  // Holds the number of navigations, which will be in one training batch
  trainingThreshold: 100,

  // Holds the number of trainings, after which the weights will be averaged
  numOfTrainings: 10,

  useConvModel: true,

  isTraining: false,

  buildModel()
  {
    // Define URL input, which has a size of 2000 chars
    // (not including batch dimension).
    const urlInput = tf.input({
      shape: [this.maxUrlLength], dtype: "float32"
    });

    let output = null;

    if (this.useConvModel)
    {
      output = this.buildConvFeatureExtractor(urlInput);
    }
    else
    {
      output = this.buildRNNFeatureExtractor(urlInput);
    }

    this.model = tf.model({inputs: urlInput, outputs: output});
    const optimizer = tf.train.adam();
    this.model.compile({loss: tf.losses.sigmoidCrossEntropy, optimizer});
  },

  buildRNNFeatureExtractor(urlInput)
  {
    const embedding = tf.layers.embedding({
      inputDim: this.maxUrlLength,
      outputDim: 48
    });

    const lstmLayer1 = tf.layers.lstm({
      dropout: 0.2,
      units: 8,
    });

    const denseLayer1 = tf.layers.dense({
      units: 64,
      dtype: "float32",
      inputDType: "float32",
      activation: "relu"
    });

    const denseLayer2 = tf.layers.dense({
      units: 1,
      dtype: "float32",
      inputDType: "float32",
      activation: "sigmoid"
    });


    const output = denseLayer2.apply(
      denseLayer1.apply(
        lstmLayer1.apply(
          embedding.apply(
            urlInput
    ))));

    return output;
  },

  buildConvFeatureExtractor(urlInput)
  {
    const embedding = tf.layers.embedding({
      inputDim: this.maxUrlLength,
      outputDim: 48
    });

    const convLayer1 = tf.layers.conv1d({
      padding: "same",
      strides: 3,
      filters: 64,
      kernelSize: 5
    });

    const convLayer2 = tf.layers.conv1d({
      padding: "same",
      strides: 2,
      filters: 128,
      kernelSize: 3
    });

    const maxPool1 = tf.layers.maxPooling1d({
      padding: "same",
      strides: 1,
      poolSize: 3
    });

    const dropoutLayer1 = tf.layers.dropout({rate: 0.2});

    const convLayer3 = tf.layers.conv1d({
      padding: "same",
      strides: 1,
      filters: 256,
      kernelSize: 1
    });

    const maxPool2 = tf.layers.maxPool1d({
      padding: "same",
      strides: 1,
      poolSize: 3
    });

    const denseLayer1 = tf.layers.dense({
      units: 128,
      dtype: "float32",
      inputDType: "float32",
      activation: "relu"
    });

    const denseLayer2 = tf.layers.dense({
      units: 1,
      dtype: "float32",
      inputDType: "float32",
      activation: "sigmoid"
    });

    const flattenLayer = tf.layers.flatten();

    const output = denseLayer2.apply(
      denseLayer1.apply(
        flattenLayer.apply(
          maxPool2.apply(
            convLayer3.apply(
              dropoutLayer1.apply(
                maxPool1.apply(
                  convLayer2.apply(
                    convLayer1.apply(
                      embedding.apply(
                        urlInput
    ))))))))));

    return output;
  },

  maxUrlLength: 256,
  maxCharCount: 1,

  resetToMeanWeights(numOfTrainings)
  {
    let curWeights = [];
    // Iterate though all stored weights
    this.intermediateWeights.forEach(element =>
    {
      let cwIndex = 0;
      // Iterate through all layers and sum the weights across layers
      element.forEach(layer =>
      {
        curWeights[cwIndex] = (typeof curWeights[cwIndex]) == "undefined" ?
          layer : curWeights[cwIndex].add(layer);
        cwIndex++;
      });
    });
    // Divide weights of each layer by the number of trainings
    curWeights.forEach(element =>
    {
      element = element.div(numOfTrainings);
    });

    this.model.setWeights(curWeights);
  },

  normalizeInputUrl(exampleUrl)
  {
    let normalizedInput = [];
    // Go over each character in example
    for (let i = 0; i < this.maxUrlLength; i++)
    {
      if (i < exampleUrl.length)
      {
        // Normalize the input dividing by the number of possible chars
        normalizedInput[i] = exampleUrl.charCodeAt(i);
      }
      else
      {
        normalizedInput[i] = 0;
      }
    }
    return normalizedInput;
  },

  async fit()
  {
    if (this.isTraining)
    {
      return;
    }

    this.isTraining = true;
  
    // encode each URL
    let normalizedInput = [];
    this.data.forEach(entry =>
    {
      normalizedInput.push(this.normalizeInputUrl(entry.url));
    });

    // Produce an input tensor for the URL
    let encodedInput = tf.tensor2d(
      normalizedInput,
      [this.data.length, this.maxUrlLength],
      "float32"
    );

    // Produce an output tensor
    let labelsFloat = new Float32Array(this.labels);
    let encodedOutput = tf.tensor2d(
      labelsFloat,
      [this.data.length, 1],
      "float32"
    );

    let h = await this.model.fit(
      encodedInput,
      encodedOutput,
      {
        epochs: 2,
        batchSize: 32
      }
    );
    let lastLossPos = h.history.loss.length - 1;
    console.log("Last loss was: " + h.history.loss[lastLossPos]);
    let memInfo = tf.memory()
    console.log("Memory usage: " +
      memInfo.numBytes + " bytes in " +
      memInfo.numTensors + " tensors");

    tf.dispose(encodedInput);
    tf.dispose(encodedOutput);
    tf.dispose(h);

    // TODO: Uncomment below for federated learning emulation
    /*
    this.intermediateWeights[this.curIteration++] = this.model.getWeights();
    if (this.curIteration >= this.numOfTrainings)
    {
      this.resetToMeanWeights(numOfTrainings);
      this.curIteration = 0;
    }
    */
    // Purge this batch and start collecting a new one
    this.clearData();

    this.isTraining = false;

    return null;
  },

  predict(location)
  {
    let encodedInput = [];
    encodedInput[0] = this.normalizeInputUrl(location);

    const inferenceResult = tf.tidy(() =>
    {
      return this.model.predict(
        tf.tensor2d(encodedInput, [1, this.maxUrlLength])
      ).dataSync()[0];
    });

    return inferenceResult;
  },

  clearData()
  {
    this.data = [];
    this.labels = [];
    this.curNavigation = 0;
  },

  addRecord(request, filter)
  {
    let isBlocked = (filter != null);
    this.data.push({
      url: request.url,
      type: request.type,
      domain: request.docDomain,
      isThirdParty: request.thirdParty,
      specific: request.specificOnly,
      label: isBlocked
    });

    this.labels.push(isBlocked ? 1 : 0);
  }
};

exports.MlModel = new MachineLearning();