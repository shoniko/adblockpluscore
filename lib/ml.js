/** @module ml */

"use strict";


const featureScaler = require("feature-scaler");
const tf = require("@tensorflow/tfjs");

/**
 * Class for Machine Learning functions
 * @constructor
 */
function MachineLearning()
{
  this.buildModel();
}

MachineLearning.prototype = {

  data: [],
  labels: [],
  model: null,

  buildModel()
  {
    // Define URL input, which has a size of 2000 chars
    // (not including batch dimension).
    const urlInput = tf.input({
      shape: [this.maxUrlLength], dtype: "int32"
    });
    const urlEmbedding = tf.layers.embedding({
      inputDim: this.maxUrlLength, outputDim: 20
    });
    const urlLstm = tf.layers.lstm({units: 8, returnSequences: true});
    const urlFeatures = urlLstm.apply(urlEmbedding.apply(urlInput));

//    const auxInput = tf.input({shape: 4});
    // This doesn work. How can we concat the features...?
//    const allFeaturesLayer = tf.layers.concatenate();
//    const allFeatures = allFeaturesLayer.apply([urlFeatures, auxInput]);
    const allFeatures = urlFeatures;

    const denseLayer1 = tf.layers.dense(
      {units: 32, activation: "relu", useBias: true, inputDim: [20]}
    );
    const denseLayer2 = tf.layers.dense(
      {units: 1, activation: "softmax", useBias: true, inputDim: [32]}
    );

    const flattenLayer = tf.layers.flatten();

    const output = denseLayer2.apply(
      denseLayer1.apply(
        flattenLayer.apply(allFeatures)
      )
    );
    this.model = tf.model({inputs: [urlInput], outputs: output});
    this.model.compile({loss: "meanSquaredError", optimizer: "sgd"});
  },

  maxUrlLength: 2000,
  maxCharCount: 256,

  stringToPaddedArray(exampleUrl)
  {
    // Convert string to array of ints and pad each to maxUrlLength
    const encodedExample = [];
    for (let i = 0; i < this.maxUrlLength; i++)
    {
      if (i >= exampleUrl.length)
      {
        encodedExample.push(0);
      }
      else
      {
        encodedExample.push(exampleUrl.charCodeAt(i));
      }
    }
    return encodedExample;
  },

  fit()
  {
    // encode each URL
    let encodedInput = [];
    for (let k = 0; k < this.data.length; k++)
    {
      encodedInput.push(this.stringToPaddedArray(this.data[k].url));
    }

    encodedInput = tf.tensor2d(
      encodedInput,
      [encodedInput.length, this.maxUrlLength],
      "int32"
    );

    this.model.fit(
      encodedInput,
      tf.tensor(this.labels),
      {batchSize: this.data.length, epochs: 1}
    );
  },

  predict(location)
  {
    let encodedInput = this.stringToPaddedArray(location);

    const inferenceResult = this.model.predict(
      tf.tensor(encodedInput)
    ).asScalar().value;
    console.log(location + " " + inferenceResult);
    return inferenceResult;
  },

  addRecord(location, typeMask, docDomain, thirdParty, specificOnly, isBlocked)
  {
    this.data.push({
      url: location,
      type: typeMask,
      domain: docDomain,
      isThirdParty: thirdParty,
      specific: specificOnly,
      label: isBlocked
    });

    this.labels.push(isBlocked);
  }
};

exports.MlModel = new MachineLearning();

browser.webNavigation.onCommitted.addListener(details =>
{
  // We retrain our ML model on every navigation
  exports.MlModel.fit();
});