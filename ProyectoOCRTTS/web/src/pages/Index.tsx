import { AIReader } from '@/components/AIReader';

const Index = () => {
  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Lector IA con OCR
          </h1>
          <p className="text-muted-foreground">
            Convierte texto e imágenes a voz con inteligencia artificial
          </p>
        </header>

        <AIReader />
      </div>
    </div>
  );
};

export default Index;
