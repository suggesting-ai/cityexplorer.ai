const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const cities = require('./cities.json');

const app = express();
app.use(bodyParser.json());

// CNN-Modell laden (Annahme: Das Modell wurde bereits trainiert und gespeichert)
let model;
async function loadModel() {
    model = await tf.loadLayersModel('file://./model/model.json');
}
loadModel();

app.post('/generate-tour', async (req, res) => {
    try {
        const { city, location, duration, interests } = req.body;

        // Eingabedaten für das CNN vorbereiten
        const cityData = cities[city];
        if (!cityData) {
            return res.status(400).json({ error: 'Stadt nicht gefunden' });
        }

        const input = tf.tensor2d([
            [
                cityData.population / 1000000, // Normalisierte Bevölkerung
                parseFloat(duration),
                interests.includes('history') ? 1 : 0,
                interests.includes('art') ? 1 : 0,
                interests.includes('food') ? 1 : 0,
                interests.includes('nature') ? 1 : 0,
                interests.includes('architecture') ? 1 : 0,
                interests.includes('shopping') ? 1 : 0
            ]
        ]);

        // CNN-Vorhersage
        const prediction = model.predict(input);
        const tourIndices = await prediction.argMax(1).data();

        // Tour basierend auf der Vorhersage generieren
        const tour = {
            center: cityData.center,
            stops: tourIndices.map(index => cityData.attractions[index])
        };

        res.json(tour);
    } catch (error) {
        console.error('Fehler bei der Tourgenerierung:', error);
        res.status(500).json({ error: 'Interner Serverfehler' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server läuft auf Port ${PORT}`);
});