package lesson01;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class ND4j_create {
    public static void main(String[] args) {
        System.out.println(Nd4j.getBackend());
        INDArray tensor1 = Nd4j.create(new double[]{1,2,3});
        INDArray tensor2 = Nd4j.create(new double[]{10.0,20.0,30.0});
        System.out.println(tensor1.add(tensor2));
    }
}
