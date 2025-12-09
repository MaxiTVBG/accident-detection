import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';

interface ReportProps {
  report: any;
  onDetectAgain: () => void;
  currentLanguage: string; // Add currentLanguage prop
}

const MotionCard = motion(Card);
const MotionTableRow = motion(TableRow);

const Report: React.FC<ReportProps> = ({ report, onDetectAgain, currentLanguage }) => {
  const { t } = useTranslation();

  if (!report) {
    return null;
  }

  if (report.error) {
    return (
      <div className="max-w-4xl mx-auto p-4">
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{report.error}</AlertDescription>
        </Alert>
        <div className="text-center mt-4">
          <Button onClick={onDetectAgain}>{t('detectAgain')}</Button>
        </div>
      </div>
    );
  }

  const {
    timestamp_readable,
    gps_coordinates,
    detection_parameters,
    ai_analysis_multilang, // Access the multi-language analysis
  } = report;

  // Select the AI analysis based on the current language, fallback to 'en'
  const ai_analysis = ai_analysis_multilang ? (ai_analysis_multilang[currentLanguage] || ai_analysis_multilang['en']) : null;


  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  const tableRowVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.5,
      },
    },
  };

  const detectionParamKeys: { [key: string]: string } = {
    "confidence_threshold_accident": "confThreshold",
    "frame_confirmation_threshold": "frameThreshold",
    "speed_threshold_for_stop": "speedThreshold",
    "iou_threshold_for_collision": "iouThreshold",
  };

  return (
    <motion.div
      className="max-w-7xl mx-auto p-4"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <motion.h1 className="text-5xl font-bold text-center tracking-tighter mb-8" variants={itemVariants}>
        {t('reportTitle')}
      </motion.h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <MotionCard variants={itemVariants} whileHover={{ y: -5, transition: { duration: 0.2 } }} className="lg:col-span-1 bg-card/50 backdrop-blur-sm border-border/20">
          <CardHeader>
            <CardTitle className="text-2xl">{t('accidentSummary')}</CardTitle>
          </CardHeader>
          <CardContent>
            <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.1 } } }}>
              <Table>
                <TableBody>
                  <MotionTableRow variants={tableRowVariants}>
                    <TableCell className="font-semibold text-muted-foreground">{t('timestamp')}</TableCell>
                    <TableCell>{timestamp_readable}</TableCell>
                  </MotionTableRow>
                  <MotionTableRow variants={tableRowVariants}>
                    <TableCell className="font-semibold text-muted-foreground">{t('location')}</TableCell>
                    <TableCell>
                      <a href={gps_coordinates.google_maps_link} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                        {t('googleMaps')}
                      </a>
                    </TableCell>
                  </MotionTableRow>
                </TableBody>
              </Table>
            </motion.div>
          </CardContent>
        </MotionCard>

        {ai_analysis && (
          <MotionCard variants={itemVariants} whileHover={{ y: -5, transition: { duration: 0.2 } }} className="lg:col-span-2 bg-card/50 backdrop-blur-sm border-border/20">
            <CardHeader>
              <CardTitle className="text-2xl">{t('aiAnalysis')}</CardTitle>
            </CardHeader>
            <CardContent>
              {ai_analysis.error ? (
                <Alert variant="destructive">
                  <AlertTitle>AI Analysis Failed</AlertTitle>
                  <AlertDescription>{ai_analysis.error}</AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-4">
                  {ai_analysis.accident_summary && ai_analysis.accident_summary.description && (
                    <div>
                      <p className="font-semibold text-muted-foreground">{t('description')}</p>
                      <p>{ai_analysis.accident_summary.description}</p>
                    </div>
                  )}
                  {ai_analysis.accident_summary && (
                    <>
                      <div>
                        <p className="font-semibold text-muted-foreground">{t('severity')}</p>
                        <p className="text-xl">{ai_analysis.accident_summary.severity}</p>
                      </div>
                      {ai_analysis.accident_summary.inferred_sequence_of_events && (
                        <div>
                          <p className="font-semibold text-muted-foreground">{t('sequence')}</p>
                          <ul className="list-disc list-inside mt-2 space-y-1">
                            {ai_analysis.accident_summary.inferred_sequence_of_events.map((event: string, index: number) => (
                              <li key={index}>{event}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </CardContent>
          </MotionCard>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {ai_analysis?.participants && (
          <MotionCard variants={itemVariants} whileHover={{ y: -5, transition: { duration: 0.2 } }} className="bg-card/50 backdrop-blur-sm border-border/20">
            <CardHeader>
              <CardTitle className="text-2xl">{t('participants')}</CardTitle>
            </CardHeader>
            <CardContent>
              <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.1 } } }}>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t('type')}</TableHead>
                      <TableHead>{t('color')}</TableHead>
                      <TableHead>{t('damage')}</TableHead>
                      <TableHead>{t('role')}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ai_analysis.participants.map((participant: any, index: number) => (
                      <MotionTableRow key={index} variants={tableRowVariants}>
                        <TableCell>{participant.type}</TableCell>
                        <TableCell>{participant.color}</TableCell>
                        <TableCell>{participant.visible_damage}</TableCell>
                        <TableCell>{participant.role}</TableCell>
                      </MotionTableRow>
                    ))}
                  </TableBody>
                </Table>
              </motion.div>
            </CardContent>
          </MotionCard>
        )}

        <MotionCard variants={itemVariants} whileHover={{ y: -5, transition: { duration: 0.2 } }} className="bg-card/50 backdrop-blur-sm border-border/20">
          <CardHeader>
            <CardTitle className="text-2xl">{t('detectionParams')}</CardTitle>
          </CardHeader>
          <CardContent>
            <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.1 } } }}>
              <Table>
                <TableBody>
                  {Object.entries(detection_parameters).map(([key, value], index) => (
                    <MotionTableRow key={index} variants={tableRowVariants}>
                      <TableCell className="font-semibold text-muted-foreground">{t(detectionParamKeys[key] || key)}</TableCell>
                      <TableCell>{String(value)}</TableCell>
                    </MotionTableRow>
                  ))}
                </TableBody>
              </Table>
            </motion.div>
          </CardContent>
        </MotionCard>
      </div>

      <motion.div className="text-center mt-12" variants={itemVariants}>
        <motion.div whileHover={{ scale: 1.05, y: -2 }} whileTap={{ scale: 0.95 }}>
          <Button onClick={onDetectAgain} size="lg" className="font-bold text-lg px-8 py-6 rounded-full shadow-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-300">
            {t('detectAgain')}
          </Button>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default Report;
