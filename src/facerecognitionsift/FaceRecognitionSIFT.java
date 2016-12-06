/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facerecognitionsift;

import java.io.File;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.print;
import org.bytedeco.javacpp.opencv_features2d.FlannBasedMatcher;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_FIND_BIGGEST_OBJECT;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_SCALE_IMAGE;
import org.bytedeco.javacpp.opencv_xfeatures2d;

/**
 *
 * @author nguyentrungtin
 */
public class FaceRecognitionSIFT {

    /**
     * @param args the command line arguments
     */
   private ArrayList<ArrayList<opencv_core.Mat>> trainData;
    private ArrayList<ArrayList<opencv_core.Mat>> testData;
    private opencv_objdetect.CascadeClassifier face_cascade;
    public FaceRecognitionSIFT(){
        this.trainData= new ArrayList<ArrayList<opencv_core.Mat>>();
        this.testData= new ArrayList<ArrayList<opencv_core.Mat>>();
        File folder = new File("");
        String fileName = folder.getAbsolutePath() + "/src/haarcascades/haarcascade_frontalface_default.xml";
        this.face_cascade = new opencv_objdetect.CascadeClassifier();
        this.face_cascade.load(fileName);
    }
    /**
     * @param args the command line arguments
     */
   
    public static void main(String[] args) {
        // TODO code application logic here
        File folder = new File("");
        FaceRecognitionSIFT fg = new FaceRecognitionSIFT();
        String fileName = folder.getAbsolutePath() + "/src/face/";
        System.out.println("Starting face recognition!");
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        opencv_xfeatures2d.SIFT sift = opencv_xfeatures2d.SIFT.create();
        int radius = 1;
        int neighbors = 16;
        for(int idx = 0; idx <  listOfFiles.length / 8; idx++){
            ArrayList<opencv_core.Mat> trt = new ArrayList<>();
            ArrayList<opencv_core.Mat> tst = new ArrayList<>();
            for (int i = 0; i < 8; i++){
                if (listOfFiles[i + idx * 8].getName().contains(".jpg")){
                    String name =  listOfFiles[idx * 8 + i].getName();
                    opencv_core.Mat image = imread(fileName + "/" + name, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                    opencv_core.Mat face = fg.faceDetector(image);
                    
                    resize(face, face, new opencv_core.Size(120, 120));
                    opencv_imgproc.GaussianBlur(face, face, new opencv_core.Size(3,3), CV_PI);
                    opencv_core.KeyPointVector kpoint = new opencv_core.KeyPointVector();
                    //Mat lbp = ELBP_(face, radius, neighbors);
                    Mat lbp = calcLBP(face);
                    //lbp.convertTo(lbp, opencv_core.CV_8U);
                    //imwrite(name, lbp);
                    Mat features = new Mat();
                    sift.detectAndCompute(lbp, new Mat(), kpoint, features);
                    //features.reshape(1, 1);
                    //opencv_core.normalize(features, features);
                    if(i < 6){
                        trt.add(features);
                    }else tst.add(features);
                }
            }
            fg.testData.add(tst);
            fg.trainData.add(trt);
        }
        fg.matchFace();
            
    }
    
     public opencv_core.Mat faceDetector(opencv_core.Mat image){
        opencv_core.RectVector objectList = new opencv_core.RectVector();
         //Only support biggest face
        face_cascade.detectMultiScale( image, objectList, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, 
                                        new opencv_core.Size(140, 140), new opencv_core.Size(image.cols(), image.cols()) );
        opencv_core.Mat face_resized = new opencv_core.Mat();
        if (objectList.size() <= 0 ){
            resize(image, image, new opencv_core.Size(120, 120));
            return image;
        }
        opencv_core.Rect maxRect = new opencv_core.Rect();
        maxRect = objectList.get(0);
        face_resized = image.apply(maxRect);
        return face_resized;
    }
     
     public void matchFace(){
        int err = 0;
        int[] label = new int[this.testData.size() * this.testData.get(0).size()];
        for(int i = 0; i < this.testData.size(); i++){
            for(int j = 0; j < this.testData.get(0).size(); j++){
                label[i* this.testData.get(0).size() + j] = i;
            }
        }
        for(int i = 0; i < this.testData.size(); i++){
            double[] a;
            for(int ele = 0; ele < this.testData.get(0).size(); ele++){
                a = this.findClass(this.testData.get(i).get(ele));
                if(a[1] != label[i * this.testData.get(0).size() + ele ])
                    err++;
                System.out.println( ele + ": " + "Predict: " + a[1] + " : " +  label[i * this.testData.get(0).size() + ele ] + " Distance: " + a[0]);
            }
            System.out.println(".......................");
        }
        System.out.println("Error : " + err + " Total: " + this.testData.size() *  this.testData.get(0).size());
        System.out.println( "Accuracy rate: " + (1 - ( err * 1.0) / (this.testData.size() * this.testData.get(0).size())));
    }
     
    public double[] findClass(opencv_core.Mat hist){
        FlannBasedMatcher matcher = new FlannBasedMatcher();
        double[] score = new double[this.trainData.size()];
        for(int i = 0; i < this.trainData.size(); i++){
            double tmp = 0;
            for(int j = 0; j < this.trainData.get(i).size(); j++){
                opencv_core.DMatchVector d = new opencv_core.DMatchVector();
                matcher.match(hist, this.trainData.get(i).get(j), d);
                tmp += findMaxMin(d);
            }
            tmp = tmp / (this.trainData.get(0).size() * 1.0);
            score[i] = tmp; 
        }
        double[] min = new double[2];
        min[0] = 10000000;
        for (int i = 0; i < score.length; i++){
            if(min[0] > score[i]){
                min[0] = score[i];
                min[1] = i;
            }
        }
        return min;
    }
    
    public double findMaxMin(opencv_core.DMatchVector d){
        double min = 10000;
        for(int i = 0; i < d.size(); i++){
            opencv_core.DMatch s = d.get(i);
            if(s.distance() < min){
                min = s.distance();
            }
        }
        return min;
    }
    
    public static double chiSquare(opencv_core.Mat img, opencv_core.Mat img1){
        float dis = 0;
        DoubleRawIndexer idx = img.createIndexer();
        DoubleRawIndexer idx1 = img1.createIndexer();
        for(int i = 0; i < img.cols(); i++){
            if((idx.get(0, i) + idx1.get(0, i)) == 0){
               dis += 0;
            }
            else{
               dis += Math.pow((idx.get(0, i) - idx1.get(0, i)), 2)/(idx.get(0, i) + idx1.get(0, i));
            }
        }
        return dis;
   }
    public static double euclideanDistance(Mat img, Mat img1){
        double score = 0;
        FloatRawIndexer idx = img.createIndexer();
        FloatRawIndexer idx1 = img1.createIndexer();
        System.out.println(img.rows() + " " + img.cols());
        System.out.println(img1.rows() + " " + img1.cols());
        
        for(int i = 0; i < img.rows(); i++){
            double tmp = 0;
            for(int j = 0; j < img.cols(); j++){
                tmp += Math.pow((idx.get(i, j) - idx1.get(i, j)), 2);
            }
            score += Math.sqrt(tmp);
        }
        score /= img.rows();
        return score;
    }
    public static Mat ELBP_(Mat src, int radius, int neighbors){
       Mat dst = new Mat(src.rows() - 2 * radius, src.cols() - 2 * radius, CV_32SC1);
        for (int n = 0; n < neighbors; n++) {
            // sample points
            float x = (float) ((radius) * Math.cos(2.0 * Math.PI * n / (neighbors)));
            float y = (float) ((radius) * - Math.sin(2.0 * Math.PI * n / (neighbors)));
            // relative indices
            int fx = (int) (Math.floor(x));
            int fy = (int) (Math.floor(y));
            int cx = (int) (Math.ceil(x));
            int cy = (int) (Math.ceil(y));
            // fractional part
            float ty = y - fy;
            float tx = x - fx;
            // set interpolation weights
            float w1 = (1 - tx) * (1 - ty);
            float w2 = tx * (1 - ty);
            float w3 = (1 - tx) * ty;
            float w4 = tx * ty;
            // iterate through your data
            UByteRawIndexer index = src.createIndexer();
            IntRawIndexer index_dst = dst.createIndexer();
            for (int i = radius; i < src.rows() - radius; i++) {
                for (int j = radius; j < src.cols() - radius; j++) {
                    float t = w1 * index.get(i + fy, j + fx) + w2 * index.get(i + fy, j + cx) + w3 * index.get(i + cy, j + fx) + w4 * index.get(i + cy, j + cx);

                    int temp = ((t > index.get(i, j) ? 1 : 0) << n);
                    index_dst.put(i - radius, j - radius, temp);
                }
            }

        }
        return dst;
    }
    public static Mat calcLBP(Mat image){
        Mat lbp = new Mat(image.rows() - 2, image.cols() - 2, image.type());
        UByteRawIndexer dst1Idx = image.createIndexer();
        UByteRawIndexer dst = lbp.createIndexer();
        int rows = image.rows(), cols = image.cols();
        for( int r = 1; r < rows - 1; r++){
            for( int c = 1; c < cols - 1; c++){
                float center = dst1Idx.get(r, c, 0);
                int code = 0;
                if(dst1Idx.get(r - 1, c - 1, 0) >= center){
                    code += 128;
                }
                if(dst1Idx.get(r - 1, c, 0) >= center){
                    code += 64;
                }
                if(dst1Idx.get(r - 1, c + 1, 0) >= center){
                    code += 32;
                }
                if(dst1Idx.get(r, c + 1, 0) >= center){
                    code += 16;
                }
                if(dst1Idx.get(r + 1, c + 1, 0) >= center){
                    code += 8;
                }
                if(dst1Idx.get(r + 1, c, 0) >= center){
                    code += 4;
                }
                if(dst1Idx.get(r + 1, c - 1, 0) >= center){
                    code += 2;
                }
                if(dst1Idx.get(r, c - 1, 0) >= center){
                    code += 1;
                }
                dst.put(r - 1, c - 1, code);
            }
        }
        return lbp;	
    }
}
