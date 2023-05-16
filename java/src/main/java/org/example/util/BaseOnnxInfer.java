package org.example.util;

import ai.onnxruntime.*;

import java.util.Arrays;
import java.util.Collections;

public abstract class BaseOnnxInfer{

    //静态加载动态链接库
    static{ nu.pattern.OpenCV.loadShared(); }

    private OrtEnvironment env;
    private String[] inputNames;
    private OrtSession[] sessions;

    /**
     * 构造函数
     * @param modelPath
     * @param threads
     */
    public BaseOnnxInfer(String modelPath, int threads){
        try {
            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setInterOpNumThreads(threads);
            opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            this.sessions = new OrtSession[]{env.createSession(modelPath, opts)};
            this.inputNames = new String[]{this.sessions[0].getInputNames().iterator().next()};
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 获取环境信息
     * @return
     */
    public OrtEnvironment getEnv() {
        return env;
    }

    /**
     * 获取输入端的名称
     * @return
     */
    public String getInputName() {
        return inputNames[0];
    }

    /**
     * 获取session
     * @return
     */
    public  OrtSession getSession() {
        return sessions[0];
    }

    /**
     * 获取输入端的名称
     * @return
     */
    public String[] getInputNames() {
        return inputNames;
    }

    /**
     * 获取session
     * @return
     */
    public OrtSession[] getSessions() {
        return sessions;
    }

    /**
     * 关闭服务
     */
    public void close(){
        try {
            if(sessions != null){
                for(OrtSession session : sessions){
                    session.close();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 将张量进行评估
     * @param transform
     * @return
     */
    public float[] predict(OnnxTensor transform){

        float[][] s;
        OrtSession.Result output = null;
        try {
            output = getSession().run(Collections.singletonMap(getInputName(), transform));
            s = (float[][]) output.get(0).getValue();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return s[0];
    }

    /**
     * 将模型处理结果的概率变现
     * @param input
     * @return
     */
    public float[] softmax(float[] input) {
        float sumExp = 0f;
        float[] output = new float[input.length];

        // Compute the sum of the exponentials of input values
        for (int i = 0; i < input.length; i++) {
            sumExp += Math.exp(input[i]);
        }

        // Compute softmax probabilities for each input value
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i]) / sumExp;
        }

        return output;
    }

    public String[] postprocess(float[] output) {
        String[] classNames = {"番茄叶斑病", "苹果黑星病", "葡萄黑腐病"};
        int numClasses = output.length;
        // 创建一个索引数组，用于排序。
        Integer[] indexes = new Integer[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            indexes[i] = i;
        }
        // 按输出向量中的分数进行排序。
        Arrays.sort(indexes, (a, b) -> Float.compare(output[b], output[a]));

        // 根据排序的顺序创建新的类名数组。
        String[] sortedClassNames = new String[numClasses];
        for (int i = 0; i < numClasses; ++i) {
            sortedClassNames[i] = classNames[indexes[i]];
        }

        return sortedClassNames;
    }

    public float[] getFloats(float[] softmax) {
        Arrays.sort(softmax);
        float[] sorted = new float[softmax.length];
        for (int i = 0; i < softmax.length; i++) {
            sorted[i] = softmax[softmax.length - i - 1];
        }
        return sorted;
    }
}
