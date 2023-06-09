package org.example.util;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Base64.Decoder;

/**
 * 图片加载工具
 */
public class ImageMat implements Serializable {

    //静态加载动态链接库
    static{ nu.pattern.OpenCV.loadShared(); }
    private OrtEnvironment env = OrtEnvironment.getEnvironment();

    //对象成员
    private Mat mat;
    private ImageMat(Mat mat){
        this.mat = mat;
    }


    /**
     * 直接读取Mat
     * @param mat 图片mat值
     * @return
     */
    public static ImageMat fromCVMat(Mat mat){
        try {
            return new ImageMat(mat);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }


    /**
     * 读取图片，并转换为Mat
     * @param inputStream 图片数据
     * @return
     */
    public static ImageMat fromInputStream(InputStream inputStream){
        try {
            BufferedImage image = ImageIO.read(inputStream);
            return fromBufferedImage(image);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    /**
     * 读取图片，并转换为Mat
     * @param image 图片数据
     * @return
     */
    public static ImageMat fromBufferedImage(BufferedImage image){
        try {
            if(image.getType() != BufferedImage.TYPE_3BYTE_BGR){
                BufferedImage temp = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
                Graphics2D g = temp.createGraphics();
                try {
                    g.setComposite(AlphaComposite.Src);
                    g.drawImage(image, 0, 0, null);
                } finally {
                    g.dispose();
                }
                image = temp;
            }
            byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            Mat mat = Mat.eye(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
            mat.put(0, 0, pixels);
            return new ImageMat(mat);
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }


    /**
     * 克隆ImageMat
     * @return
     */
    public ImageMat clone(){
        return ImageMat.fromCVMat(this.mat.clone());
    }


    /**
     * 重新设置图片尺寸,不释放原始图片数据
     * @param width     图片宽度
     * @param height    图片高度
     * @return
     */
    public ImageMat resizeAndNoReleaseMat(int width, int height){
        return this.resize(width, height, false);
    }


    /**
     * 重新设置图片尺寸
     * @param width     图片宽度
     * @param height    图片高度
     * @param release   是否释放参数mat
     * @return
     */
    private ImageMat resize(int width, int height, boolean release){
        try {
            Mat dst = new Mat();
            Imgproc.resize(mat, dst, new Size(width,height), 0, 0, Imgproc.INTER_LINEAR);
            return new ImageMat(dst);
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 对图像进行预处理,并释放原始图片数据
     * @param scale     图像各通道数值的缩放比例
     * @param mean      用于各通道减去的值，以降低光照的影响
     * @param swapRB    交换RB通道，默认为False.
     * @return
     */
    public ImageMat blobFromImageAndDoReleaseMat(double scale, Scalar mean, boolean swapRB){
        return this.blobFromImage(scale, mean, swapRB, true);
    }

    /**
     * 对图像进行预处理
     * @param scale     图像各通道数值的缩放比例
     * @param mean      用于各通道减去的值，以降低光照的影响
     * @param swapRB    交换RB通道，默认为False.
     * @param release   是否释放参数mat
     * @return
     */
    private ImageMat blobFromImage(double scale, Scalar mean, boolean swapRB, boolean release){
        try {
            Mat dst = Dnn.blobFromImage(mat, scale, new Size( mat.cols(), mat.rows()), mean, swapRB);
            java.util.List<Mat> mats = new ArrayList<>();
            Dnn.imagesFromBlob(dst, mats);
            dst.release();
            return new ImageMat(mats.get(0));
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 转换为整形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public int[][][][] to4dIntArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dIntArray(firstChannel, false);
    }

    /**
     * 转换为整形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public int[][][][] to4dIntArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dIntArray(firstChannel, true);
    }


    /**
     * 转换为整形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private int[][][][] to4dIntArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            int[][][][] array;
            if(firstChannel){
                array = new int[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = (int) Math.round(c[k]);
                        }
                    }
                }
            }else{
                array = new int[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = (int) Math.round(c[k]);
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为长整形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public long[][][][] to4dLongArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dLongArray(firstChannel, false);
    }

    /**
     * 转换为长整形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public long[][][][] to4dLongArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dLongArray(firstChannel, true);
    }
    /**
     * 转换为长整形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private long[][][][] to4dLongArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            long[][][][] array;
            if(firstChannel){
                array = new long[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = Math.round(c[k]);
                        }
                    }
                }
            }else{
                array = new long[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = Math.round(c[k]);
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为单精度形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public float[][][][] to4dFloatArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dFloatArray(firstChannel, false);
    }

    /**
     * 转换为单精度形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public float[][][][] to4dFloatArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dFloatArray(firstChannel, true);
    }

    /**
     * 转换为单精度形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private float[][][][] to4dFloatArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            float[][][][] array;
            if(firstChannel){
                array = new float[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = (float) c[k];
                        }
                    }
                }
            }else{
                array = new float[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = (float) c[k];
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }


    /**
     * 转换为双精度形数组,不释放原始图片数据
     * @param firstChannel
     * @return
     */
    public double[][][][] to4dDoubleArrayAndNoReleaseMat(boolean firstChannel){
        return this.to4dDoubleArray(firstChannel, false);
    }

    /**
     * 转换为双精度形数组,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public double[][][][] to4dDoubleArrayAndDoReleaseMat(boolean firstChannel){
        return this.to4dDoubleArray(firstChannel, true);
    }

    /**
     * 转换为双精度形数组
     * @param firstChannel
     * @param release   是否释放参数mat
     * @return
     */
    private double[][][][] to4dDoubleArray(boolean firstChannel, boolean release){
        try {
            int width = this.mat.cols();
            int height = this.mat.rows();
            int channel = this.mat.channels();
            double[][][][] array;
            if(firstChannel){
                array = new double[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][k][i][j] = c[k];
                        }
                    }
                }
            }else{
                array = new double[1][height][width][channel];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        double[] c = mat.get(i, j);
                        for(int k=0; k< channel; k++){
                            array[0][i][j][k] = c[k];
                        }
                    }
                }
            }
            return array;
        }finally {
            if(release){
                this.release();
            }
        }
    }

    /**
     * 转换为单精度形OnnxTensor,并释放原始图片数据
     * @param firstChannel
     * @return
     */
    public OnnxTensor to4dFloatOnnxTensorAndDoReleaseMat(boolean firstChannel) {
        try {
            return OnnxTensor.createTensor(env, this.to4dFloatArrayAndDoReleaseMat(firstChannel));
        }catch (Exception e){
            throw new RuntimeException(e);
        }
    }
    /**
     * 释放资源
     */
    public void release(){
        if(this.mat != null){
            this.mat.release();
            this.mat = null;
        }
    }

    public OnnxTensor transform(BufferedImage img){
        float[][][][] pixelData = new float[1][3][224][224];
        // 获取图像的 RGB 像素值并进行归一化
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                int rgb = img.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = (rgb & 0xFF);
                pixelData[0][0][i][j] = (float) (Math.round(Math.min(Math.max(((float)r / 255.0f - 0.4737f) / 0.1920f, 0), 1) * 10000) / 10000.0);
                pixelData[0][1][i][j] = (float) (Math.round(Math.min(Math.max(((float)g / 255.0f - 0.4948f) / 0.1592f, 0), 1) * 10000) / 10000.0);
                pixelData[0][2][i][j] = (float) (Math.round(Math.min(Math.max(((float)b / 255.0f - 0.4336f) / 0.2184f, 0), 1) * 10000) / 10000.0);
            }
        }
        OnnxTensor inputTensor = null;
        try {
            inputTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), pixelData);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return inputTensor;
    }
}
