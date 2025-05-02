const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");
const { InferenceSession, Tensor } = require("onnxruntime-node");
const { createCanvas, loadImage } = require("canvas");
const uuid = require("uuid");
const https = require("https"); 

const app = express();
const PORT = 3000;

const sslOptions = {
  key: fs.readFileSync(path.join(__dirname, 'certs', 'key.pem')),
  cert: fs.readFileSync(path.join(__dirname, 'certs', 'cert.pem'))
};
app.use(cors());

const MODEL_PATH = path.join(__dirname, "resnet18_model.onnx");
const TEMP_DIR = path.join(__dirname, "temp");
const admin = require('firebase-admin');
const serviceAccount = require("D:/STUDY/6th_sem/SGP-III/SERVICE-KEY/shotalert-605f1-c9addc7f66f9.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const firestore = admin.firestore();
async function sendGunfireAlert(userId) {
  try {
    // First, get the username of the sender (person who detected gunfire)
    const userDoc = await firestore.collection('users').doc(userId).get();
    
    if (!userDoc.exists) {
      console.log(`❌ User document not found for ID: ${userId}`);
      return;
    }
    
    // Get the user's name or email (fallback) to display in notifications
    const userData = userDoc.data();
    const senderName = userData.displayName || userData.email || userData.name || "A contact";
    
    // 1. Get contacts of user
    const contactsSnapshot = await firestore.collection('users')
      .doc(userId)
      .collection('contacts')
      .get();

    const fcmTokens = [];

    // 2. For each contact, get their user document by phone number (assuming phone is unique)
    for (const doc of contactsSnapshot.docs) {
      const contactPhone = doc.data().phone;
      if (!contactPhone)
        {
          console.log(`⚠️ Contact missing phone number. Skipping.`);

          continue;
        }
        console.log(`🔍 Looking for user with phone: ${contactPhone}`);

      // Search user by phone
      const userQuery = await firestore.collection('users')
        .where('mobile', '==', contactPhone)
        .limit(1)
        .get();

      if (!userQuery.empty) {
        const contactUser = userQuery.docs[0];
        const fcmToken = contactUser.data().fcmToken;
        if (fcmToken) {
          console.log(`✅ Found FCM token for ${contactPhone}: ${fcmToken}`);

          fcmTokens.push(fcmToken);
        }
        else
        {
          console.log(`❌ User ${contactPhone} found, but no FCM token.`);

        }
      }
      else{
        console.log(`❌ No user found with phone: ${contactPhone}`);
      }
    }

    if (fcmTokens.length > 0) {
      // Send to all contacts with username included
      await admin.messaging().sendEachForMulticast({
        tokens: fcmTokens,
        notification: {
          title: '🚨 Gunfire Detected!',
          body: `${senderName} detected gunfire nearby. Stay alert!`,
        },
        data: {
          type: 'gunfire_alert',
          senderName: senderName, // Also including in data for app-side use
        },
      });
      
      console.log('Notifications sent successfully with sender name.');
    } else {
      console.log('No FCM tokens found for contacts.');
    }
  } catch (error) {
    console.error('Error sending notifications:', error);
  }
}

if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR);
}

let ortSession;
(async () => {
  try {
    ortSession = await InferenceSession.create(MODEL_PATH);
    console.log("✅ ONNX model loaded successfully");
  } catch (error) {
    console.error(`❌ Failed to load ONNX model: ${error.message}`);
  }
})();

const upload = multer({ dest: TEMP_DIR });

const generateSpectrogram = (audioPath, spectrogramPath) => {
  return new Promise((resolve, reject) => {
    exec(`python generate_spectogram.py "${audioPath}" "${spectrogramPath}"`, (error, stdout, stderr) => {
      if (error) {
        console.error(`❌ Python Error: ${error.message}`);
        reject(error);
      } else {
        console.log("✅ Spectrogram generated via Python");
        resolve();
      }
    });
  });
};

app.get("/", (req, res) => {
  res.json({
    message: "Welcome to the Gun Sound Detection API. Use POST /predict with an audio file.",
  });
});

app.post("/predict", upload.single("file"), async (req, res) => {
  
  const userId = req.body.userId; // <-- ✅ Fetch the userId

  if (!userId) {
    return res.status(400).json({ error: "User ID is required" });
  }
  
  if (!ortSession) {
    return res.status(500).json({ error: "ONNX model not initialized" });
  }

  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const audioFilePath = req.file.path;
  const spectrogramPath = path.join(TEMP_DIR, `spectrogram_${uuid.v4()}.png`);

  try {
    console.log(`🔄 Received file: ${req.file.originalname}`);

    await generateSpectrogram(audioFilePath, spectrogramPath);

    const image = await loadImage(spectrogramPath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, 224, 224);

    const imgData = ctx.getImageData(0, 0, 224, 224).data;
    const inputTensor = new Float32Array(224 * 224 * 3);

    let tensorIndex = 0;
    for (let i = 0; i < imgData.length; i += 4) {
      inputTensor[tensorIndex++] = imgData[i] / 255.0;     // R
      inputTensor[tensorIndex++] = imgData[i + 1] / 255.0; // G
      inputTensor[tensorIndex++] = imgData[i + 2] / 255.0; // B
    }

    const input = {
      [ortSession.inputNames[0]]: new Tensor('float32', inputTensor, [1, 3, 224, 224]),
    };

    const results = await ortSession.run(input);
    const output = results[ortSession.outputNames[0]];
    const outputData = output.data;

    // const predldx = outputData.indexOf(Math.max(.. .outputData));
    // const confidence = Math.max(.. .outputData) * 100;
    // Apply softmax to normalize the output
    function softmax(arr) {
      const expValues = arr.map((x) => Math.exp(x));
      const sumExpValues = expValues.reduce((a, b) => a + b, 0);
      return expValues.map((x) => x / sumExpValues);
    }

    const softmaxOutput = softmax(outputData);
    const predIdx = softmaxOutput.indexOf(Math.max(...softmaxOutput));
    const confidence = Math.max(...softmaxOutput) * 100;

    const classLabels = { 0: "Non-Gun", 1: "Gun" };
    const predictionLabel = classLabels[predIdx] || "Cannot determine";

    console.log(`✅ Prediction: ${predictionLabel}, Confidence: ${confidence.toFixed(2)}%`);

    fs.unlinkSync(audioFilePath);
    fs.unlinkSync(spectrogramPath);

      // 🧠 IF gunfire is detected, send alert!
  if (predictionLabel === "Gun") {
    await sendGunfireAlert(userId); // <-- 🔥 Call your notification function
  }


    res.status(200).json({
      prediction: predictionLabel,
      confidence: `${confidence.toFixed(2)}%`,

    });

  } catch (error) {
    console.error(`❌ Prediction error: ${error.message}`);
    if (fs.existsSync(audioFilePath)) fs.unlinkSync(audioFilePath);
    if (fs.existsSync(spectrogramPath)) fs.unlinkSync(spectrogramPath);

    res.status(500).json({ error: "An error occurred while processing the file" });
  }
});

// app.listen(PORT, () => {
//   console.log(`🚀 Server is running at http://localhost:${PORT}`);
// });

https.createServer(sslOptions, app).listen(PORT, () => {
  console.log(`🚀 HTTPS Server is running at https://localhost:${PORT}`);
});
// 192.168.217.211