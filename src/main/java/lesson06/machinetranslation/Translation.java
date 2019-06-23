package lesson06.machinetranslation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.FilteredSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class Translation {
    public static String enPath = "SMT/testCorpus.en";
    public static String frPath = "SMT/testCorpus.fr";
    public static String testEnPath = "SMT/predictCorpus.en";

    public static final int VOCAB_SIZE = 100;
    public static final int FEATURE_VEC_SIZE = 64;
    
    private static SequenceToSequenceIterator getTrainData(int batchSize, int vocabSize) throws Exception {

        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> en = getIterator(true, vocabSize);
        Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> fr = getIterator(false, vocabSize);

        return new SequenceToSequenceIterator(batchSize, vocabSize, en.getFirst(), fr.getFirst(), en.getSecond(), fr.getSecond());
    }

    private static Pair<SequenceIterator<VocabWord>, AbstractCache<VocabWord>> getIterator(boolean english, int maxVocabSize) throws Exception {
        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
        File file;
        if (english) file = new File(enPath);
        else file = new File(frPath);
        
        /*define line reader,tokenizer for initalizing SentenceTransformer*/
        /*-----------------------------------*/
        BasicLineIterator lineIter = new BasicLineIterator(file);       //read file line one by one
        TokenizerFactory t = new DefaultTokenizerFactory();             //default tokenizer(for english)
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                                                                .iterator(lineIter)
                                                                .tokenizerFactory(t)
                                                                .build();

        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(transformer)
                                                                   .build();
        /*finish defining SentenceTransformer*/
        /*-----------------------------------*/
        /*build vacabulary*/
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                                                        .addSource(sequenceIterator, 1)
                                                        .setTargetVocabCache(vocabCache)
                                                        .setEntriesLimit(maxVocabSize)
                                                        .build();
        constructor.buildJointVocabulary(false, true);
        sequenceIterator.reset();
        /*finish building vacabulary*/
        SequenceIterator<VocabWord> filteredIterator = new FilteredSequenceIterator<>(sequenceIterator, vocabCache);
        return new Pair<>(filteredIterator, vocabCache);
    }

    private static class SequenceToSequenceIterator implements MultiDataSetIterator {

        /**
         * 
         */
        private static final long serialVersionUID = -649505930720554358L;
        
        private int batchSize;
        private int vocabSize;
        private SequenceIterator<VocabWord> iter1;
        private SequenceIterator<VocabWord> iter2;
        private AbstractCache<VocabWord> vocabCache1;
        private AbstractCache<VocabWord> vocabCache2;
        
        private boolean toTestSet;

        public SequenceToSequenceIterator(int batchSize, int vocabSize, SequenceIterator<VocabWord> iter1, SequenceIterator<VocabWord> iter2,
                                          AbstractCache<VocabWord> vocabCache1, AbstractCache<VocabWord> vocabCache2) {
            this.batchSize = batchSize;
            this.vocabSize = vocabSize;
            this.iter1 = iter1;
            this.iter2 = iter2;
            this.vocabCache1 = vocabCache1;
            this.vocabCache2 = vocabCache2;
        }

        public AbstractCache<VocabWord> getVocabCache1() {
            return vocabCache1;
        }

        public AbstractCache<VocabWord> getVocabCache2() {
            return vocabCache2;
        }

        /***
         *Generate Test Data 
         */
        public MultiDataSet generateTest(int testSize) {
            toTestSet = true;
            MultiDataSet testData = next(testSize);
            return testData;
        }
        
        @Override
        public MultiDataSet next(int num) {
            
            if( toTestSet ){
                reset();
                batchSize = num;
            }
            
            /*--Contruct Input Sequence--*/
            List<List<VocabWord>> iter1List = new ArrayList<>(batchSize);   //Innner List means each train sentence
                                                                            //List of List store a @batchSize train data
            
            for (int i = 0; i < batchSize && iter1.hasMoreSequences(); i++) {
                iter1List.add(iter1.nextSequence().getElements());
            }
            /*--Finish Contructing Input Sequence--*/
            /*-------------------------------------*/
            /*--Contruct Output Sequence--*/
            List<List<VocabWord>> iter2List = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && iter2.hasMoreSequences(); i++) {
                iter2List.add(iter2.nextSequence().getElements());
            }
            /*--Finish Contructing Output Sequence--*/
            /*--------------------------------------*/
            int numExamples = Math.min(iter1List.size(), iter2List.size()); //ensure input/output have same number
            int in1Length = 0;
            int in2Length = 0;
            /*reserve maximum capacity of input/output sentence*/
            for (int i = 0; i < numExamples; i++) {
                in1Length = Math.max(in1Length, iter1List.get(i).size());
            }
            for (int i = 0; i < numExamples; i++) {
                in2Length = Math.max(in2Length, iter2List.get(i).size());
            }
            /*finish reserving maximum capacity of input/output sentence*/
            /*--------------------------------------*/
            //2 inputs here, and 1 output
            //First input: a sequence of word indexes for iter1 words
            //Second input: a sequence of word indexes for iter2 words (shifted by 1, with an additional 'go' class as first time step)
            //Output: sequence of word indexes for iter2 words (with an additional 'stop' class as the last time step)
            //Also need mask arrays

            INDArray in1 = Nd4j.create(numExamples, 1, in1Length);  //32*39 Matrix, 32:batchSize, 39:max lenghth of 32 sentence
                                                                    //data format for each row(haffman index for each word):
                                                                    //283.0 128.0 ... 10.0 0.0 0.0 ... 0.0
            INDArray in1Mask = Nd4j.ones(numExamples, in1Length);   //32*39 Matrix, 
                                                                    //data format for each row(if the position is active):
                                                                    //1.0 1.0 ... 1.0 0.0 0.0 ... 0.0
            int[] arr1 = new int[3];
            int[] arr2 = new int[2];
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter1List.get(i);
                arr1[0] = i;        //arr(0) store the index of batch sentence
                arr2[0] = i;

                int j = 0;                          //index of the word in the sentence
                for (VocabWord vw : list) {         //traverse the list which store an entire sentence
                    arr1[2] = j++;
                    in1.putScalar(arr1, vw.getIndex());
                }
                for (; j < in1Length; j++) {
                    arr2[1] = j;
                    in1Mask.putScalar(arr2, 0.0);
                }
            }
            /*Almost the same as the first input matrix, the only difference is that:
             * 1. the first element of in2 is equal to vacabulary size
             * 2. the last element of in2mask is 1.0
             * All of above is a convention*/
            INDArray in2 = Nd4j.create(numExamples, 1, in2Length + 1);
            INDArray in2Mask = Nd4j.ones(numExamples, in2Length + 1);
            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter2List.get(i);
                arr1[0] = i;
                arr2[0] = i;

                //First time step: "go" index = vocab size (as word indexes are 0 to vocabSize-1 inclusive)
                arr1[2] = 0;
                in2.putScalar(arr1, vocabSize);

                int j = 1;
                for (VocabWord vw : list) {
                    arr1[2] = j++;
                    in2.putScalar(arr1, vw.getIndex());
                }
                for (; j < in2Length; j++) {    //last element is 1.0,maybe a bug?
                    arr2[1] = j;
                    in2Mask.putScalar(arr2, 0.0);
                }
            }

            //Using a one-hot representation here. Can't use indexes line for input
            INDArray out = Nd4j.create(numExamples, vocabSize + 1, in2Length + 1);
            INDArray outMask = Nd4j.ones(numExamples, in2Length + 1);

            for (int i = 0; i < numExamples; i++) {
                List<VocabWord> list = iter2List.get(i);
                arr1[0] = i;
                arr2[0] = i;

                int j = 0;
                for (VocabWord vw : list) {
                    arr1[1] = vw.getIndex();
                    arr1[2] = j++;
                    out.putScalar(arr1, 1.0);   //one hot representation
                }

                //Last time step: "stop" index = vocab size (as word indexes are 0 to vocabSize-1 inclusive)
                arr1[1] = vocabSize;
                arr1[2] = j++;
                out.putScalar(arr1, 1.0);

                for (; j < in2Length; j++) {
                    arr2[1] = j;
                    outMask.putScalar(arr2, 0.0);
                }
            }
            
            INDArray[] inputs = new INDArray[]{in1, in2};
            INDArray[] inputMasks = new INDArray[]{in1Mask, in2Mask};
            INDArray[] labels = new INDArray[]{out};
            INDArray[] labelMasks = new INDArray[]{outMask};

            return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

        }

        @Override
        public void reset() {
            iter1.reset();
            iter2.reset();
        }

        @Override
        public boolean hasNext() {
            return iter1.hasMoreSequences() && iter2.hasMoreSequences();
        }

        @Override
        public MultiDataSet next() {
            return next(batchSize);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported");
        }

        @Override
        public boolean resetSupported() {
            // TODO Auto-generated method stub
            return true;
        }

        @Override
        public boolean asyncSupported() {
            // TODO Auto-generated method stub
            return true;
        }

        @Override
        public MultiDataSetPreProcessor getPreProcessor() {
            // TODO Auto-generated method stub
            return null;
        }
    }
    
    public static ComputationGraphConfiguration getSMTModel(final int VOCAB_ENCODER_SIZE, final int VOCAB_DECODER_SIZE){
        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .seed(123456L)
            .l2(0.0001)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.01))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .graphBuilder()
            .addInputs("inputLine", "decoderInput")
            .setInputTypes(InputType.recurrent(VOCAB_ENCODER_SIZE), InputType.recurrent(VOCAB_DECODER_SIZE))
            //
            .addLayer("embeddingEncoder",new EmbeddingLayer.Builder().nIn(VOCAB_ENCODER_SIZE + 1).nOut(256).build(),"inputLine")
            .addLayer("encoder",new LSTM.Builder().nIn(256).nOut(256).activation(Activation.SOFTSIGN).build(),"embeddingEncoder")
            .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
            .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
            //
            .addLayer("embeddingDecoder", new EmbeddingLayer.Builder().nIn(VOCAB_DECODER_SIZE + 1).nOut(256).activation(Activation.IDENTITY).build(),"decoderInput")
            .addVertex("embeddingDecoderSeq", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "embeddingDecoder")
            //
            .addVertex("merge", new MergeVertex(), "embeddingDecoderSeq", "dup")
            .addLayer("decoder",new LSTM.Builder().nIn(256 + 256).nOut(256).activation(Activation.SOFTSIGN).build(),"merge")
            .addLayer("output",new RnnOutputLayer.Builder().nIn(256).nOut(VOCAB_DECODER_SIZE + 1).activation(Activation.SOFTMAX).build(),"decoder")
            .setOutputs("output")
            .build();
        return configuration;
    }
    
    private static String predict(ComputationGraph net, AbstractCache<VocabWord> frCache, INDArray en){
        StringBuilder sbPredictRes = new StringBuilder();
        //
        int nPredictWord = 1;
        INDArray curFr, preFr = null;
        String lastPredictWord = null;
        while( true ){
            if( nPredictWord > 100 )break;
            curFr = Nd4j.create(1, 1, nPredictWord);
            if( nPredictWord == 1 ){
                double[] newdoubles = new double[nPredictWord];
                newdoubles[0] = VOCAB_SIZE;
                INDArray newRow = Nd4j.create(newdoubles);  
                curFr.putRow(0, newRow);
                preFr = Nd4j.create(1, 1, nPredictWord);
                Nd4j.copy(curFr, preFr);
            }else{
                double indexofLastWord = (double)frCache.indexOf(lastPredictWord);
                INDArray tempRow = preFr.getRow(0);
                double[] doubles = tempRow.data().asDouble();
                double[] newdoubles = new double[doubles.length + 1];
                System.arraycopy(doubles, 0, newdoubles, 0, doubles.length);
                newdoubles[newdoubles.length - 1] = indexofLastWord;
                INDArray newRow = Nd4j.create(newdoubles);
                curFr.putRow(0, newRow);
                preFr = Nd4j.create(1, 1, nPredictWord);
                Nd4j.copy(curFr, preFr);
            }
            INDArray tempRes = net.outputSingle(false, new INDArray[]{en, curFr});
            INDArray tempArgMaxRes1 = Nd4j.argMax(tempRes,1);
            int wordIndex = (int)tempArgMaxRes1.getDouble(nPredictWord - 1);
            if( wordIndex == VOCAB_SIZE )break;
            lastPredictWord = frCache.wordAtIndex(wordIndex);
            sbPredictRes.append(lastPredictWord + " ");
            //
            nPredictWord++;
        }
        //
        return sbPredictRes.toString();
    }
    
    private static INDArray getTestData(AbstractCache<VocabWord> enCache) throws Exception{
        
        File file = new File(testEnPath);
        FileInputStream fis= new FileInputStream(file);
        InputStreamReader isr=new InputStreamReader(fis);
        BufferedReader br=new BufferedReader(isr);
        String line;
        String[] words = new String[]{};
        while( (line = br.readLine()) != null ){
            words = line.toLowerCase().split(" ");
        }
        INDArray ret = Nd4j.create(1, 1, words.length);
        double[] doubles = new double[words.length];
        for( int index = 0; index < doubles.length; ++index ){
            doubles[index] = enCache.indexOf(words[index]);
        }
        INDArray newRow = Nd4j.create(doubles);
        ret.putRow(0, newRow);
        br.close();
        return ret;
    }
    
    public static void main(String[] args) throws Exception {
        SequenceToSequenceIterator train = getTrainData(32, VOCAB_SIZE);
        
        AbstractCache<VocabWord> enCache = train.getVocabCache1();
        
        AbstractCache<VocabWord> frCache = train.getVocabCache2();

        ComputationGraphConfiguration netconf = getSMTModel(VOCAB_SIZE,VOCAB_SIZE);
        ComputationGraph net = new ComputationGraph(netconf);
        net.init();
        
        net.setListeners(new ScoreIterationListener(1));
        for( int nEpoch = 0; nEpoch < 100; ++nEpoch ){
            net.fit(train);
            System.out.println("Finish Epoch: " + nEpoch);
            System.out.println("\nDONE");
            INDArray testEn = getTestData(enCache);
            System.out.println(predict(net, frCache, testEn));
        }
        //
        System.out.println("\nDONE");
        INDArray testEn = getTestData(enCache);
        System.out.println(predict(net, frCache, testEn));

    }

}
