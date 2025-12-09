import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Report from './Report';
import Live from './Live';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';

const MotionPath = motion.path;
const MotionPolyline = motion.polyline;
const MotionLine = motion.line;

const UploadIcon = () => {
  const iconVariants = {
    hidden: { pathLength: 0, opacity: 0 },
    visible: (i: number) => {
      const delay = i * 0.5;
      return {
        pathLength: 1,
        opacity: 1,
        transition: {
          pathLength: { delay, type: "spring", duration: 1.5, bounce: 0 },
          opacity: { delay, duration: 0.01 },
        },
      };
    },
  };

  return (
    <motion.svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-16 w-16 text-muted-foreground"
      initial="hidden"
      animate="visible"
    >
      <MotionPath d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" custom={0} variants={iconVariants} />
      <MotionPolyline points="17 8 12 3 7 8" custom={1} variants={iconVariants} />
      <MotionLine x1="12" x2="12" y1="3" y2="15" custom={2} variants={iconVariants} />
    </motion.svg>
  );
};

const LoadingIndicator = () => {
    const { t } = useTranslation();
    return (
        <motion.div
            className="flex flex-col items-center gap-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
        >
            <motion.div className="w-24 h-24 relative">
            {[...Array(3)].map((_, i) => (
                <motion.div
                key={i}
                className="absolute w-full h-full border-4 border-primary rounded-full"
                initial={{ rotate: 0, scale: 0, opacity: 0 }}
                animate={{
                    rotate: 360,
                    scale: [0, 1, 0.8, 1],
                    opacity: [0, 1, 1, 0],
                }}
                transition={{
                    duration: 2,
                    ease: "easeInOut",
                    repeat: Infinity,
                    repeatDelay: 1,
                    delay: i * 0.3,
                }}
                />
            ))}
            </motion.div>
            <p className="text-xl font-semibold mt-4">{t('analyzing')}</p>
        </motion.div>
    );
};


function App() {
  const { t, i18n } = useTranslation();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [detectionMode, setDetectionMode] = useState<'upload' | 'live'>('upload');


  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setUploadedFile(file);
    setLoading(true);
    setError(null);
    setReport(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const uploadResponse = await axios.post(`http://localhost:8000/upload-video?lang=${i18n.language}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (uploadResponse.data.report_filename) {
        const reportFilename = uploadResponse.data.report_filename;
        const pollReport = async () => {
          try {
            const reportResponse = await axios.get(`http://localhost:8000/reports/${reportFilename}`);
            if (reportResponse.data) {
              setReport(reportResponse.data);
              setLoading(false);
            } else {
              setTimeout(pollReport, 2000);
            }
          } catch (error) {
            console.error('Error fetching report:', error);
            setTimeout(pollReport, 2000);
          }
        };
        pollReport();
      } else {
        setError(t('reportFail'));
        setLoading(false);
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setError(t('processingError'));
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'video/*': [] } });

  const handleDetectAgain = () => {
    setReport(null);
    setUploadedFile(null);
    setError(null);
  };

  const renderContent = () => {
    if (detectionMode === 'live') {
      return <Live language={i18n.language} />;
    }

    if (report) {
      return (
         <motion.div
            key="report"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
            className="w-full"
          >
            <Report report={report} onDetectAgain={handleDetectAgain} currentLanguage={i18n.language} />
          </motion.div>
      )
    }

    return (
       <motion.div
            key="upload"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="w-full max-w-2xl"
          >
            <Card className="w-full text-center bg-card/50 backdrop-blur-sm border-border/20 shadow-2xl">
              <CardHeader className="py-8">
                <CardTitle className="text-4xl font-extrabold tracking-tight">{t('appTitle')}</CardTitle>
                <CardDescription className="text-lg text-muted-foreground pt-2">
                  {t('appDescription')}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  {...getRootProps()}
                  className={`relative border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-300 ease-in-out ${isDragActive ? 'border-primary bg-accent/30' : 'border-border/50'}`}
                >
                  <input {...getInputProps()} />
                  {loading ? (
                    <div className="flex flex-col items-center gap-4">
                      <LoadingIndicator />
                      {uploadedFile && <p className="text-muted-foreground mt-4">{uploadedFile.name}</p>}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-4">
                      <UploadIcon />
                      <p className="text-xl font-semibold mt-4">{isDragActive ? t('dropIt') : t('dragAndDrop')}</p>
                      <p className="text-muted-foreground">{t('orClick')}</p>
                    </div>
                  )}
                </div>

                {error && (
                  <Alert variant="destructive" className="mt-6 text-left bg-destructive/20 border-destructive/50">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </motion.div>
    )
  }

  const changeLanguage = async (lng: string) => {
    await i18n.changeLanguage(lng);
    // No need to call translate-report endpoint anymore, as the report now contains all languages
    // The Report component will handle displaying the correct language based on i18n.language
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
        <div className="absolute top-4 right-4 flex gap-2">
            <Button variant={i18n.language === 'en' ? 'secondary' : 'ghost'} onClick={() => changeLanguage('en')}>EN</Button>
            <Button variant={i18n.language === 'bg' ? 'secondary' : 'ghost'} onClick={() => changeLanguage('bg')}>BG</Button>
        </div>
       <div className="mb-8 flex gap-4 bg-muted p-2 rounded-lg">
        <Button 
          variant={detectionMode === 'upload' ? 'secondary' : 'ghost'} 
          onClick={() => setDetectionMode('upload')}
          className="transition-all"
        >
          {t('uploadVideo')}
        </Button>
        <Button 
          variant={detectionMode === 'live' ? 'secondary' : 'ghost'} 
          onClick={() => setDetectionMode('live')}
          className="transition-all"
        >
          {t('liveDetection')}
        </Button>
      </div>
      <AnimatePresence mode="wait">
       {renderContent()}
      </AnimatePresence>
    </div>
  );
}

export default App;
