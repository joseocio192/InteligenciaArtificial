import { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

export const FaceEnrollment = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [userName, setUserName] = useState('');
  const [capturedDescriptors, setCapturedDescriptors] = useState<Float32Array[]>([]);
  const [enrolledUsers, setEnrolledUsers] = useState<Record<string, number[][]>>({});
  const [enrollmentMode, setEnrollmentMode] = useState<'upload' | 'webcam' | null>(null);
  const [uploadedImages, setUploadedImages] = useState<string[]>([]);

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
        loadExistingEmbeddings();
        startVideo();
      } catch (err) {
        console.error('Error loading face-api.js models:', err);
        setError('Error cargando modelos de face-api.js.');
        setIsLoading(false);
      }
    };

    loadModels();
  }, []);

  const loadExistingEmbeddings = async () => {
    try {
      const response = await fetch('/models/face_embeddings_faceapi.json');
      if (response.ok) {
        const data = await response.json();
        setEnrolledUsers(data);
        console.log('Loaded existing embeddings:', Object.keys(data));
      }
    } catch (err) {
      console.log('No existing embeddings found, starting fresh');
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const imageUrls: string[] = [];
    const newDescriptors: Float32Array[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const imageUrl = URL.createObjectURL(file);
      imageUrls.push(imageUrl);

      try {
        const img = await faceapi.fetchImage(imageUrl);
        const detection = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detection) {
          newDescriptors.push(detection.descriptor);
        } else {
          console.warn(`No se detectó rostro en la imagen: ${file.name}`);
        }
      } catch (err) {
        console.error(`Error procesando imagen ${file.name}:`, err);
      }
    }

    setUploadedImages(prev => [...prev, ...imageUrls]);
    setCapturedDescriptors(prev => [...prev, ...newDescriptors]);
    
    alert(`Se procesaron ${files.length} imágenes. Se detectaron ${newDescriptors.length} rostros.`);
  };

  const startVideo = () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Tu navegador no soporta acceso a la cámara.');
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
        setError('No se pudo acceder a la cámara.');
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
      }
    };

    setInterval(detectFaces, 100);
  };

  const captureDescriptor = async () => {
    const video = videoRef.current;
    if (!video || !modelsLoaded) return;

    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection) {
      setCapturedDescriptors(prev => [...prev, detection.descriptor]);
      alert(`Capturado ${capturedDescriptors.length + 1} descriptor(es). Captura 5-10 para mejor precisión.`);
    } else {
      alert('No se detectó ningún rostro. Intenta de nuevo.');
    }
  };

  const enrollUser = () => {
    if (!userName.trim()) {
      alert('Por favor ingresa un nombre de usuario.');
      return;
    }

    if (capturedDescriptors.length === 0) {
      alert('Captura al menos un descriptor antes de registrar.');
      return;
    }

    // Convertir Float32Array a arrays normales para JSON
    const descriptorsAsArrays = capturedDescriptors.map(desc => Array.from(desc));
    
    const updatedUsers = {
      ...enrolledUsers,
      [userName]: descriptorsAsArrays
    };

    setEnrolledUsers(updatedUsers);
    
    // Mostrar JSON para copiar
    console.log('=== COPIA ESTE JSON Y GUÁRDALO COMO face_embeddings_faceapi.json ===');
    console.log(JSON.stringify(updatedUsers, null, 2));
    
    alert(`Usuario "${userName}" registrado con ${capturedDescriptors.length} descriptor(es).\n\nRevisa la consola para copiar el JSON completo y guardarlo en /public/models/face_embeddings_faceapi.json`);
    
    // Resetear
    setCapturedDescriptors([]);
    setUserName('');
    setUploadedImages([]);
    setEnrollmentMode(null);
  };

  const downloadJSON = () => {
    const dataStr = JSON.stringify(enrolledUsers, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'face_embeddings_faceapi.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Registrar Usuarios para Reconocimiento Facial</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {isLoading && !error && (
          <Alert>
            <AlertDescription>Cargando modelos...</AlertDescription>
          </Alert>
        )}

        {/* Selección del modo de registro */}
        {!enrollmentMode && !isLoading && !error && (
          <div className="space-y-4">
            <Alert>
              <AlertDescription>
                <strong>¿Cómo deseas registrar al usuario?</strong>
              </AlertDescription>
            </Alert>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button 
                onClick={() => setEnrollmentMode('upload')} 
                variant="outline"
                className="h-auto py-6 flex-col gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <div className="text-center">
                  <div className="font-semibold">Subir Fotos</div>
                  <div className="text-sm text-muted-foreground">Usa imágenes existentes</div>
                </div>
              </Button>
              <Button 
                onClick={() => {
                  setEnrollmentMode('webcam');
                  startVideo();
                }} 
                variant="outline"
                className="h-auto py-6 flex-col gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <div className="text-center">
                  <div className="font-semibold">Usar Cámara Web</div>
                  <div className="text-sm text-muted-foreground">Captura en tiempo real</div>
                </div>
              </Button>
            </div>
          </div>
        )}

        {/* Modo de subida de fotos */}
        {enrollmentMode === 'upload' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Nombre del usuario:</label>
              <Input
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                placeholder="Ej: Juan Pérez"
                className="max-w-md"
              />
            </div>

            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageUpload}
                className="hidden"
              />
              <Button 
                onClick={() => fileInputRef.current?.click()}
                disabled={!modelsLoaded}
              >
                Seleccionar Fotos ({capturedDescriptors.length} rostros detectados)
              </Button>
            </div>

            {uploadedImages.length > 0 && (
              <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
                {uploadedImages.map((url, idx) => (
                  <img 
                    key={idx} 
                    src={url} 
                    alt={`Foto ${idx + 1}`}
                    className="w-full h-24 object-cover rounded border"
                  />
                ))}
              </div>
            )}

            <div className="flex gap-2 flex-wrap">
              <Button onClick={enrollUser} disabled={capturedDescriptors.length === 0} variant="default">
                Registrar Usuario
              </Button>
              <Button onClick={() => {
                setEnrollmentMode(null);
                setCapturedDescriptors([]);
                setUploadedImages([]);
                setUserName('');
              }} variant="outline">
                Cambiar Método
              </Button>
            </div>

            <Alert>
              <AlertDescription>
                <strong>Instrucciones:</strong>
                <ol className="list-decimal ml-4 mt-2 space-y-1">
                  <li>Ingresa el nombre del usuario</li>
                  <li>Haz clic en "Seleccionar Fotos" y elige 5-10 imágenes del usuario desde diferentes ángulos</li>
                  <li>Espera a que se procesen las imágenes</li>
                  <li>Haz clic en "Registrar Usuario"</li>
                  <li>Repite para más usuarios</li>
                </ol>
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Modo de cámara web */}
        {enrollmentMode === 'webcam' && (
          <div className="space-y-4">
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

            <div>
              <label className="block text-sm font-medium mb-2">Nombre del usuario:</label>
              <Input
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                placeholder="Ej: Juan Pérez"
                className="max-w-md"
              />
            </div>

            <div className="flex gap-2 flex-wrap">
              <Button onClick={captureDescriptor} disabled={!modelsLoaded || isLoading}>
                Capturar Rostro ({capturedDescriptors.length})
              </Button>
              <Button onClick={enrollUser} disabled={capturedDescriptors.length === 0} variant="default">
                Registrar Usuario
              </Button>
              <Button onClick={() => {
                setEnrollmentMode(null);
                setCapturedDescriptors([]);
                setUserName('');
                // Detener la cámara
                const stream = videoRef.current?.srcObject as MediaStream;
                stream?.getTracks().forEach(track => track.stop());
              }} variant="outline">
                Cambiar Método
              </Button>
            </div>

            <Alert>
              <AlertDescription>
                <strong>Instrucciones:</strong>
                <ol className="list-decimal ml-4 mt-2 space-y-1">
                  <li>Ingresa el nombre del usuario</li>
                  <li>Haz clic en "Capturar Rostro" 5-10 veces desde diferentes ángulos</li>
                  <li>Haz clic en "Registrar Usuario"</li>
                  <li>Repite para más usuarios</li>
                </ol>
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Botón de descarga y usuarios registrados */}
        {enrollmentMode && (
          <div className="space-y-4 pt-4 border-t">
            <Button onClick={downloadJSON} disabled={Object.keys(enrolledUsers).length === 0} variant="outline" className="w-full">
              Descargar JSON
            </Button>

            <Alert>
              <AlertDescription className="text-sm">
                Una vez que hayas registrado todos los usuarios, haz clic en "Descargar JSON" y guarda el archivo como <code>face_embeddings_faceapi.json</code> en <code>/public/models/</code>
              </AlertDescription>
            </Alert>
          </div>
        )}

        {Object.keys(enrolledUsers).length > 0 && (
          <Alert className="bg-green-50 border-green-200">
            <AlertDescription className="text-green-800">
              <strong>Usuarios registrados:</strong> {Object.keys(enrolledUsers).join(', ')}
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};
