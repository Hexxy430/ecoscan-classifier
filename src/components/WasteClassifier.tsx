import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Upload, Camera, Loader2, Recycle, Trash2, Leaf } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Declare global tflite from CDN
declare global {
  interface Window {
    tflite: {
      loadTFLiteModel: (url: string) => Promise<any>;
    };
  }
}

interface ClassificationResult {
  label: string;
  confidence: number;
  icon: typeof Recycle | typeof Trash2 | typeof Leaf;
  color: string;
}

const WASTE_CATEGORIES = [
  { label: 'Biodegradable', icon: Leaf, color: 'text-success' },
  { label: 'Non-Biodegradable', icon: Trash2, color: 'text-destructive' },
  { label: 'Recycled', icon: Recycle, color: 'text-secondary' }
];

export default function WasteClassifier() {
  const [model, setModel] = useState<any>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    loadModel();
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const loadModel = async () => {
    try {
      setIsModelLoading(true);
      
      // Wait for tflite to be available from CDN
      let attempts = 0;
      while (!window.tflite && attempts < 50) {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }
      
      if (!window.tflite) {
        throw new Error('TFLite library failed to load');
      }
      
      const loadedModel = await window.tflite.loadTFLiteModel('/waste_classifier_f16_local.tflite');
      setModel(loadedModel);
      toast({
        title: "Model loaded successfully",
        description: "Ready to classify waste items",
      });
    } catch (error) {
      console.error('Error loading model:', error);
      toast({
        title: "Error loading model",
        description: "Please refresh the page and try again",
        variant: "destructive",
      });
    } finally {
      setIsModelLoading(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target?.result as string);
        setResult(null);
        stopCamera();
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        setImageSrc(null);
        setResult(null);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast({
        title: "Camera access denied",
        description: "Please allow camera access to use this feature",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        setImageSrc(imageData);
        stopCamera();
      }
    }
  };

  const runPrediction = async () => {
    if (!model || !imageSrc) return;

    setIsPredicting(true);
    try {
      const img = new Image();
      img.src = imageSrc;
      await new Promise((resolve) => { img.onload = resolve; });

      // Preprocess image
      let tensor = tf.browser.fromPixels(img);
      tensor = tf.image.resizeBilinear(tensor, [224, 224]);
      tensor = tensor.expandDims(0);
      tensor = tensor.cast('float32').div(255.0);

      // Run prediction
      const prediction = model.predict(tensor) as tf.Tensor;
      const probabilities = await prediction.data();
      tensor.dispose();
      prediction.dispose();

      // Get highest probability
      const maxIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
      const confidence = probabilities[maxIndex];
      
      const category = WASTE_CATEGORIES[maxIndex % WASTE_CATEGORIES.length];
      setResult({
        label: category.label,
        confidence: confidence,
        icon: category.icon,
        color: category.color
      });

      toast({
        title: "Classification complete",
        description: `${category.label} detected with ${(confidence * 100).toFixed(1)}% confidence`,
      });
    } catch (error) {
      console.error('Error during prediction:', error);
      toast({
        title: "Prediction failed",
        description: "Please try again with a different image",
        variant: "destructive",
      });
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-muted/30 to-background py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-2xl mb-4">
            <Recycle className="w-12 h-12 text-primary" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent">
            EcoScan: Waste Classifier
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload or capture an image of waste to classify it into the right category
          </p>
        </div>

        {/* Model Loading Status */}
        {isModelLoading && (
          <Card className="p-6 mb-6 bg-card/50 backdrop-blur border-primary/20">
            <div className="flex items-center justify-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <span className="text-foreground">Loading AI model...</span>
            </div>
          </Card>
        )}

        {/* Input Controls */}
        <Card className="p-8 mb-6 bg-card/80 backdrop-blur shadow-lg border-border/50">
          <div className="grid md:grid-cols-2 gap-4">
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={isModelLoading || isCameraActive}
              size="lg"
              variant="outline"
              className="h-24 flex-col gap-2 hover:bg-primary/5 hover:border-primary transition-all"
            >
              <Upload className="w-8 h-8" />
              <span>Upload Image</span>
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
            
            <Button
              onClick={isCameraActive ? capturePhoto : startCamera}
              disabled={isModelLoading}
              size="lg"
              variant="outline"
              className="h-24 flex-col gap-2 hover:bg-secondary/5 hover:border-secondary transition-all"
            >
              <Camera className="w-8 h-8" />
              <span>{isCameraActive ? 'Capture Photo' : 'Use Camera'}</span>
            </Button>
          </div>
        </Card>

        {/* Camera View */}
        {isCameraActive && (
          <Card className="p-4 mb-6 bg-card/80 backdrop-blur">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full rounded-lg"
            />
          </Card>
        )}
        <canvas ref={canvasRef} className="hidden" />

        {/* Image Preview & Classification */}
        {imageSrc && (
          <Card className="p-8 bg-card/80 backdrop-blur shadow-lg border-border/50">
            <div className="space-y-6">
              <img
                src={imageSrc}
                alt="Waste item"
                className="w-full max-h-96 object-contain rounded-lg shadow-md"
              />
              
              <Button
                onClick={runPrediction}
                disabled={!model || isPredicting}
                size="lg"
                className="w-full bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all shadow-lg"
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Classify Waste'
                )}
              </Button>

              {/* Results */}
              {result && (
                <div className="mt-6 p-6 bg-muted/50 rounded-xl border-2 border-primary/20 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-full bg-background ${result.color}`}>
                      <result.icon className="w-8 h-8" />
                    </div>
                    <div className="flex-1 space-y-3">
                      <div>
                        <h3 className="text-xl font-semibold text-foreground mb-1">
                          Classification Result
                        </h3>
                        <p className={`text-2xl font-bold ${result.color}`}>
                          {result.label}
                        </p>
                      </div>
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm text-muted-foreground">Confidence</span>
                          <span className="text-sm font-semibold text-foreground">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-1000 ease-out"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {!result && !isPredicting && (
                <div className="text-center py-8 text-muted-foreground">
                  <p>Click "Classify Waste" to analyze this image</p>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* Initial State */}
        {!imageSrc && !isCameraActive && (
          <Card className="p-12 text-center bg-card/50 backdrop-blur border-dashed border-2 border-muted-foreground/30">
            <Recycle className="w-16 h-16 mx-auto mb-4 text-muted-foreground/50" />
            <p className="text-lg text-muted-foreground">
              Waiting for waste item photo...
            </p>
            <p className="text-sm text-muted-foreground/70 mt-2">
              Upload an image or use your camera to get started
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
