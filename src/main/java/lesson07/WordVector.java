package lesson07;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.List;

public class WordVector {
    private static final Logger log = LoggerFactory.getLogger(WordVector.class);
    
    public static void main(String[] args) throws FileNotFoundException {
        String filePath = "corpus.txt";
        log.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        
        List<String> stopWords = Arrays.asList(",","\\《 ","\\》", "\\“", "。“","？”","？","：“");
        Word2Vec vec = new Word2Vec.Builder()
                        .useHierarchicSoftmax(false)
                        .negativeSample(1.0)
                        .minWordFrequency(5)
                        .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                        .stopWords(stopWords)
                        .iterations(10)
                        .layerSize(128)
                        .seed(42)
                        .windowSize(5)
                        .iterate(iter)
                        .tokenizerFactory(t)
                        .build();
        log.info("Fitting Word2Vec model....");
        vec.fit();
        log.info("Closest Words:");
        System.out.println(vec.wordsNearestSum("孙少平", 5));
        System.out.println(vec.wordsNearestSum("孙少安", 5));
        System.out.println(vec.wordsNearestSum("田晓霞", 5));
        System.out.println(vec.wordsNearestSum("田润叶", 5));
        System.out.println(vec.wordsNearestSum("白面", 5));

    }

}
