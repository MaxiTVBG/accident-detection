
import { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';

interface LiveProps {
    language: string;
}

const Live = ({ language }: LiveProps) => {
    const { t } = useTranslation();
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [socket, setSocket] = useState<WebSocket | null>(null);
    const [detectionResult, setDetectionResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [confirmedAccident, setConfirmedAccident] = useState<any>(null);

    useEffect(() => {
        if (isDetecting) {
            const ws = new WebSocket(`ws://localhost:8000/ws/live-detection`); // Language is handled by backend now
            setSocket(ws);
            
            ws.onopen = () => {
                console.log("WebSocket connection established");
                startSendingFrames();
            };
            
            ws.onmessage = (event) => {
                const result = JSON.parse(event.data);
                setDetectionResult(result);
                if (result.accident_confirmed && result.report) {
                    setConfirmedAccident(result.report);
                }
            };
            
            ws.onclose = () => {
                console.log("WebSocket connection closed");
            };

            ws.onerror = (error) => {
                console.error("WebSocket error:", error);
                setError(t('serverDisconnect'));
                setIsDetecting(false);
            };

            return () => {
                ws.close();
            };
        }
    }, [isDetecting, t]); // Removed language from dependency array as it's not directly used here for WS URL


    const startSendingFrames = () => {
        const video = videoRef.current;
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (video && context && socket?.readyState === WebSocket.OPEN) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const sendFrame = () => {
                if (socket?.readyState !== WebSocket.OPEN) return;
                context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
                const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
                socket.send(dataUrl);
                requestAnimationFrame(sendFrame);
            };
            sendFrame();
        }
    };

    const drawDetections = () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (canvas && video && detectionResult?.boxes) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                canvas.width = video.clientWidth;
                canvas.height = video.clientHeight;
                const scaleX = canvas.width / video.videoWidth;
                const scaleY = canvas.height / video.videoHeight;

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                detectionResult.boxes.forEach((box: any) => {
                    const [x1, y1, x2, y2, conf, cls] = box;
                    ctx.strokeStyle = cls === 'vehicle_incident' ? 'red' : 'green';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
                    ctx.fillStyle = cls === 'vehicle_incident' ? 'red' : 'green';
                    ctx.fillText(`${cls} (${conf.toFixed(2)})`, x1 * scaleX, y1 * scaleY - 5);
                });
            }
        }
    };

    useEffect(() => {
        if (isDetecting) {
            drawDetections();
        }
    }, [detectionResult, isDetecting]);


    const handleStartDetection = async () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
                setIsDetecting(true);
                setConfirmedAccident(null);
                setError(null);
            }
            catch (err) {
                console.error("Error accessing webcam:", err);
                setError(t('webcamError'));
            }
        }
    };

    const handleStopDetection = () => {
        setIsDetecting(false);
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        if (socket) {
            socket.close();
        }
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx?.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    // Extract AI analysis for the current language from the confirmedAccident report
    const currentAiAnalysis = confirmedAccident?.ai_analysis_multilang
        ? (confirmedAccident.ai_analysis_multilang[language] || confirmedAccident.ai_analysis_multilang['en'])
        : null;
    
    return (
        <Card className="w-full max-w-4xl text-center">
            <CardHeader>
                <CardTitle className="text-3xl font-bold">{t('liveTitle')}</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="relative aspect-video bg-muted rounded-lg overflow-hidden border">
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
                    <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
                     <AnimatePresence>
                        {detectionResult?.potential_accident && !detectionResult?.cooldown &&(
                            <motion.div 
                                initial={{opacity: 0}}
                                animate={{opacity: 1}}
                                exit={{opacity: 0}}
                                className="absolute top-4 right-4 bg-orange-500/80 text-white p-2 rounded-md">
                                {t('potentialAccident')}
                            </motion.div>
                        )}
                        {detectionResult?.cooldown && (
                             <motion.div 
                                initial={{opacity: 0}}
                                animate={{opacity: 1}}
                                exit={{opacity: 0}}
                                className="absolute top-4 right-4 bg-red-600/90 text-white font-bold p-3 rounded-lg shadow-lg">
                                {t('accidentConfirmed')}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                <div className="mt-4 flex justify-center gap-4">
                    {!isDetecting ? (
                        <Button onClick={handleStartDetection} size="lg">{t('startDetection')}</Button>
                    ) : (
                        <Button onClick={handleStopDetection} variant="destructive" size="lg">{t('stopDetection')}</Button>
                    )}
                </div>
                {error && (
                    <Alert variant="destructive" className="mt-4">
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}
                 {currentAiAnalysis && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-6 text-left p-4 border rounded-lg bg-card"
                    >
                       <h3 className="text-xl font-bold text-red-500 mb-2">{t('reportGenerated')}</h3>
                       <pre className="bg-muted p-2 rounded-sm text-sm overflow-auto">
                           {JSON.stringify(currentAiAnalysis, null, 2)}
                       </pre>
                    </motion.div>
                )}
            </CardContent>
        </Card>
    );
};

export default Live;
