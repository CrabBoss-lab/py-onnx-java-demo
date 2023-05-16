package org.example;

import ai.onnxruntime.OnnxTensor;
import org.example.util.BaseOnnxInfer;
import org.example.util.ImageMat;
import org.opencv.core.Scalar;
import java.io.*;


/**
 * Hello world!
 *
 */
public class App extends BaseOnnxInfer
{
    /**
     * 构造函数
     *
     * @param modelPath
     * @param threads
     */
    public App(String modelPath, int threads) {
        super(modelPath, threads);
    }

    public static void main(String[] args ) {
        //参数2 线程数
        App app = new App("D:\\应用缓存文件\\QQ\\model(3).onnx", 1);
        app.run();
    }

    public void run(){
        //进行预测的图片
        File file = new File("D:\\Study_need\\IMGS\\onnx\\1\\番茄叶斑病 (25).JPG");
        ImageMat imageMat = null;
        try {
            InputStream inputStream = new  FileInputStream(file);
            imageMat = ImageMat.fromInputStream(inputStream);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        ImageMat imageMats = imageMat.clone();
        //transforms.Resize((224, 224))
        OnnxTensor onnxTensor = imageMats.resizeAndNoReleaseMat(224, 224)      //将图片进行resize
                // transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                .blobFromImageAndDoReleaseMat(1.0 / 127.5, new Scalar(127.5, 127.5, 127.5), true) //进行归一化标准化
                //transforms.ToTensor(),  img_tensor = transform(img).unsqueeze(0)
                .to4dFloatOnnxTensorAndDoReleaseMat(true);  // 转为tensor  图像预处理并转换为形状为(1, C, H, W)的张量
        //y = self.net(img)
        float[] predict = this.predict(onnxTensor); //进行预测
        //p_y = torch.nn.functional.softmax(y, dim=1)
        float[] softmax = this.softmax(predict);    //计算输出的概率分布
        float[] floats = this.getFloats(softmax);   //计算输出的概率分布
        //cls_name = label_map[cls_index]
        String[] postprocess = this.postprocess(predict);         //获取预测类别对应的类别名称
        System.out.println("预测结果: " + postprocess[0]);
        System.out.println("置信度: "+ floats[0]);

    }


}
