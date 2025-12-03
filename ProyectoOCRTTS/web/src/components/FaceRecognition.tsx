import { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

export const FaceRecognition = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [faceMatcher, setFaceMatcher] = useState<faceapi.FaceMatcher | null>(null);
  const [recognizedPerson, setRecognizedPerson] = useState<string | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_URL = '/models';
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        ]);
        setModelsLoaded(true);
        await loadFaceEmbeddings();
        startVideo();
      } catch (err) {
        console.error('Error loading face-api.js models:', err);
        setError('Error cargando modelos de face-api.js. Asegúrate de tener los modelos en /public/models/');
        setIsLoading(false);
      }
    };

    loadModels();
  }, []);

  const loadFaceEmbeddings = async () => {
    try {
      // Intentar cargar primero el archivo generado con face-api.js
      let response = await fetch('/models/face_embeddings_faceapi.json');
      
      if (response.ok) {
        const data = await response.json();
        
        // Formato: { "nombre": [[...], [...]], "nombre2": [[...]] }
        const labeledDescriptors = Object.entries(data).map(([name, embeddings]) => {
          const descriptors = (embeddings as number[][]).map(emb => new Float32Array(emb));
          console.log(`Loaded ${descriptors.length} descriptors for ${name}`);
          return new faceapi.LabeledFaceDescriptors(name, descriptors);
        });
        
        const matcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
        setFaceMatcher(matcher);
        console.log(`Loaded ${labeledDescriptors.length} known faces for recognition (face-api.js format)`);
        return;
      }
      
      // Si no existe, intentar con el formato antiguo de Python
      response = await fetch('/models/face_embeddings.json');
      
      if (!response.ok) {
        console.warn('No se encontró ningún archivo de embeddings. Usa FaceEnrollment para crear uno.');
        return;
      }
      
      const data = await response.json();
      
      // Verificar si el formato es el esperado con arrays paralelos (Python format)
      if (data.embeddings && data.labels && Array.isArray(data.embeddings) && Array.isArray(data.labels)) {
        console.warn('ADVERTENCIA: Estás usando embeddings de Python que son incompatibles con face-api.js.');
        console.warn('El reconocimiento probablemente no funcionará correctamente.');
        console.warn('Por favor, usa el componente FaceEnrollment para generar embeddings compatibles.');
        
        const embeddings = data.embeddings as number[][];
        const labels = data.labels as string[];
        
        if (embeddings.length !== labels.length) {
          console.error('Mismatch between embeddings and labels length');
          return;
        }
        
        const embeddingsByLabel = new Map<string, Float32Array[]>();
        
        labels.forEach((label, idx) => {
          if (!embeddingsByLabel.has(label)) {
            embeddingsByLabel.set(label, []);
          }
          
          const embedding = embeddings[idx];
          if (embedding.length === 128) {
            embeddingsByLabel.get(label)!.push(new Float32Array(embedding));
          }
        });
        
        const labeledDescriptors = Array.from(embeddingsByLabel.entries()).map(([name, descriptors]) => {
          console.log(`Loaded ${descriptors.length} descriptors for ${name} (Python format - may not work)`);
          return new faceapi.LabeledFaceDescriptors(name, descriptors);
        });
        
        const matcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
        setFaceMatcher(matcher);
        
        console.log(`Loaded ${labeledDescriptors.length} known faces (Python format - incompatible)`);
      }
    } catch (err) {
      console.error('Error loading face embeddings:', err);
      console.warn('Continuando sin reconocimiento de usuarios.');
    }
  };

  const startVideo = () => {
    // Check if MediaDevices API is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Tu navegador no soporta acceso a la cámara o la página no está servida por HTTPS. Por favor, usa un navegador moderno y asegúrate de que la página esté en HTTPS.');
      setIsLoading(false);
      return;
    }

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error('Error accessing camera:', err);
        let errorMessage = 'No se pudo acceder a la cámara. ';
        
        if (err.name === 'NotAllowedError') {
          errorMessage += 'Por favor, permite el acceso a la cámara en tu navegador.';
        } else if (err.name === 'NotFoundError') {
          errorMessage += 'No se encontró ninguna cámara en tu dispositivo.';
        } else if (err.name === 'NotReadableError') {
          errorMessage += 'La cámara está siendo usada por otra aplicación.';
        } else {
          errorMessage += 'Error desconocido al acceder a la cámara.';
        }
        
        setError(errorMessage);
        setIsLoading(false);
      });
  };

  const handleVideoPlay = () => {
    setIsLoading(false);
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !modelsLoaded) return;

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    const detectFaces = async () => {
      if (!video || !canvas) return;

      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        
        // Reconocer rostros si tenemos un FaceMatcher
        if (faceMatcher && detections.length > 0) {
          const results = resizedDetections.map(d => {
            const bestMatch = faceMatcher.findBestMatch(d.descriptor);
            return bestMatch;
          });
          
          // Dibujar etiquetas con los nombres reconocidos
          results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { 
              label: result.toString(),
              boxColor: result.label === 'unknown' ? 'red' : 'green'
            });
            drawBox.draw(canvas);
            
            // Actualizar el nombre de la persona reconocida
            if (result.label !== 'unknown') {
              setRecognizedPerson(result.label);
            } else {
              setRecognizedPerson(null);
            }
          });
        }
      }
    };

    setInterval(detectFaces, 100);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sección 1: Reconocimiento Facial</CardTitle>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {isLoading && !error && (
          <Alert className="mb-4">
            <AlertDescription>Cargando modelos y cámara...</AlertDescription>
          </Alert>
        )}
        {recognizedPerson && (
          <Alert className="mb-4 bg-green-50 border-green-200">
            <AlertDescription className="text-green-800">
              Usuario reconocido: <strong>{recognizedPerson}</strong>
            </AlertDescription>
          </Alert>
        )}
        <div className="relative inline-block">
          <video
            ref={videoRef}
            autoPlay
            muted
            onPlay={handleVideoPlay}
            className="rounded-lg border border-border w-full max-w-2xl"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0"
          />
        </div>
      </CardContent>
    </Card>
  );
};
