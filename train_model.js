const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Trainingsdaten laden (Dies wäre in der Realität ein viel größerer Datensatz)
const trainingData = JSON.parse(fs.readFileSync('training_data.json', 'utf8'));

// Daten vorbereiten
const inputs = trainingData.map(item => [
    item.population / 1000000,
    item.duration,
    item.interests.history ? 1 : 0,
    item.interests.art ? 1 : 0,
    item.interests.food ? 1 : 0,
    item.interests.nature ? 1 : 0,
    item.interests.architecture ? 1 : 0,
    item.interests.shopping ? 1 : 0
]);

const outputs = trainingData.map(item => item.tour);

const inputTensor = tf.tensor2d(inputs);
const outputTensor = tf.tensor2d(outputs);

// Modell definieren
const model = tf.sequential();
model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [8] }));
model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
model.add(tf.layers.dense({ units: 5, activation: 'softmax' }));

// Modell kompilieren
model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

// Modell trainieren
async function trainModel() {
    await model.fit(inputTensor, outputTensor, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    // Modell speichern
    await model.save('file://./model');
    console.log('Modell wurde gespeichert');
}

trainModel();