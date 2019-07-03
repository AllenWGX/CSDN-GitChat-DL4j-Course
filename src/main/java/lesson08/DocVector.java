package lesson08;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collection;

public class DocVector {
    private static final Logger log = LoggerFactory.getLogger(DocVector.class);
    
    public static void main(String[] args) throws FileNotFoundException {
        File file = new File("corpus.txt");
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");
        
        ParagraphVectors vec = new ParagraphVectors.Builder()
                        .minWordFrequency(1)
                        .iterations(20)
                        .epochs(1)
                        .layerSize(128)
                        .learningRate(0.01)
                        .labelsSource(source)
                        .windowSize(8)
                        .workers(8)
                        .iterate(iter)
                        .useAdaGrad(true)
                        .sequenceLearningAlgorithm(new DBOW<VocabWord>())
                        .negativeSample(1.0)
                        .trainWordVectors(false)
                        .trainSequencesRepresentation(true)
                        .vocabCache(cache)
                        .tokenizerFactory(t)
                        .sampling(0)
                        .build();
        vec.fit();
        
        Collection<String> docs = vec.nearestLabels("李向前  闭住  眼睛  ，  让  汹涌  的  泪水  在  脸颊  上  溪流  般地  纵情  流淌 ", 5);
        for( String doc : docs ){
            final int docID = Integer.parseInt(doc.split("_")[1]);
            int id = 0;
            LabelAwareIterator labelIter = vec.getLabelAwareIterator();
            labelIter.reset();
            while( labelIter.hasNextDocument() ){
                LabelledDocument next = labelIter.nextDocument();
                if( id == docID  ){
                    System.out.println("similar sentence: ");
                    System.out.println(next.getContent());
                    break;
                }else{
                    ++id;
                }
            }
        }

    }

}
