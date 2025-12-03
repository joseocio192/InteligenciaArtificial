import { useState, useRef } from 'react';
import { createWorker } from 'tesseract.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';

export const AIReader = () => {
  const [textInput, setTextInput] = useState('');
  const [status, setStatus] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const generateAndPlayAudio = async (text: string) => {
    if (!text.trim()) {
      toast({
        title: "Error",
        description: "Por favor ingresa o extrae texto primero",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setStatus('Generando audio con IA...');

    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setStatus('Reproduciendo audio...');
        toast({
          title: "Éxito",
          description: "Audio generado y reproduciendo",
        });
      }
    } catch (error) {
      setStatus('Error al generar audio');
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Error al conectar con la API de TTS",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReadText = () => {
    generateAndPlayAudio(textInput);
  };

  const handleReadImage = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      toast({
        title: "Error",
        description: "Por favor selecciona una imagen",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setStatus('Extrayendo texto de la imagen...');

    try {
      const worker = await createWorker('spa');
      const { data } = await worker.recognize(file);
      await worker.terminate();

      const extractedText = data.text.trim();
      if (!extractedText) {
        setStatus('No se detectó texto en la imagen');
        toast({
          title: "Aviso",
          description: "No se encontró texto en la imagen",
        });
        setIsProcessing(false);
        return;
      }

      setTextInput(extractedText);
      setStatus('Texto extraído. Generando audio...');
      toast({
        title: "Texto extraído",
        description: "Ahora generando audio...",
      });
      
      await generateAndPlayAudio(extractedText);
    } catch (error) {
      setStatus('Error al procesar la imagen');
      toast({
        title: "Error",
        description: "Error al extraer texto de la imagen",
        variant: "destructive"
      });
      setIsProcessing(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sección 2: Lector IA (Voz de Modelo)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label htmlFor="textInput" className="block text-sm font-medium mb-2 text-foreground">
            Ingresa texto para convertir a voz:
          </label>
          <Textarea
            id="textInput"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Escribe aquí el texto que quieres escuchar..."
            className="min-h-[120px]"
            disabled={isProcessing}
          />
          <Button
            onClick={handleReadText}
            disabled={isProcessing}
            className="mt-2 w-full sm:w-auto"
          >
            Generar y Leer
          </Button>
        </div>

        <div className="border-t border-border pt-4">
          <label htmlFor="imageInput" className="block text-sm font-medium mb-2 text-foreground">
            O sube una imagen para extraer texto:
          </label>
          <Input
            ref={fileInputRef}
            id="imageInput"
            type="file"
            accept="image/*"
            disabled={isProcessing}
            className="mb-2"
          />
          <Button
            onClick={handleReadImage}
            disabled={isProcessing}
            variant="secondary"
            className="w-full sm:w-auto"
          >
            Leer Imagen
          </Button>
        </div>

        {status && (
          <Alert>
            <AlertDescription>{status}</AlertDescription>
          </Alert>
        )}

        <div className="pt-4">
          <label className="block text-sm font-medium mb-2 text-foreground">
            Reproductor de Audio:
          </label>
          <audio
            ref={audioRef}
            controls
            className="w-full"
            onEnded={() => setStatus('Audio finalizado')}
          />
        </div>
      </CardContent>
    </Card>
  );
};
