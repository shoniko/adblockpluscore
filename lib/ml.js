/** @module ml */

"use strict";

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
    const tf = require("@tensorflow/tfjs");

    // Define URL input, which has a size of 2000 chars
    // (not including batch dimension).
    const urlInput = tf.input({shape: [2000]});
    const urlEmbedding = tf.layers.embedding({inputDim: 2000, outputDim: 79});
    const urlLstm = tf.layers.lstm({units: 8, returnSequences: true});
    const urlFeatures = urlLstm.apply(urlEmbedding.apply(urlInput));

    const auxInput = tf.input({shape: 4});
    // This doesn work. How can we concat the features...?
    const allFeaturesLayer = tf.layers.concatenate();
    const allFeatures = allFeaturesLayer.apply([urlFeatures, auxInput]);

    const denseLayer1 = tf.layers.dense(
      {units: 64, activation: "relu", useBias: true}
    );
    const denseLayer2 = tf.layers.dense(
      {units: 64, activation: "relu", useBias: true}
    );
    const denseLayer3 = tf.layers.dense(
      {units: 64, activation: "relu", useBias: true}
    );

    const output = denseLayer3.apply(
      denseLayer2.apply(
        denseLayer1.apply(
          allFeatures)));
    this.model = tf.model({inputs: [urlInput, auxInput], outputs: output});
  },

  fit()
  {
    const featureScaler = require("feature-scaler");
    let encodedInput = featureScaler.encode(this.data);

    const h = this.model.fit(encodedInput, this.labels, 
      {batchSize: 4, epochs: 1});
    console.log(encodedInput);
    console.log(h);
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

