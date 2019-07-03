package lesson07;


import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class GloVeExample {
	private static final Logger log = LoggerFactory.getLogger(GloVeExample.class);

    public static void main(String[] args) throws Exception {
        File inputFile = new File("corpus.txt");

        // creating SentenceIterator wrapping our training corpus
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> stopWords = Arrays.asList(",","\\《 ","\\》", "\\“", "。“","？”","？","：“");
        Glove glove = new Glove.Builder()
        		.layerSize(512)
                .iterate(iter)
                .iterations(1)
                .tokenizerFactory(t)
                .alpha(0.75)
                .learningRate(0.1)
                .minWordFrequency(5)
                .stopWords(stopWords)
                .useAdaGrad(true)
                .workers(8)
                .epochs(1)
                // cutoff for weighting function
                .xMax(100)
                // training is done in batches taken from training corpus
                .batchSize(1000)

                // if set to true, batches will be shuffled before training
                .shuffle(true)

                // if set to true word pairs will be built in both directions, LTR and RTL
                .symmetric(true)

                .build();

        glove.fit();
        
        System.out.println(glove.wordsNearest("孙少平", 5));
        System.out.println(glove.wordsNearest("孙少安", 5));
        System.out.println(glove.wordsNearest("田晓霞", 5));

        System.exit(0);
    }
}
