package lesson08;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.graph.data.GraphLoader;
import org.deeplearning4j.graph.graph.Graph;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class GraphEmbeding {
	private static final String delim = "\t";
	private static String format = "%s:%s\n";
	
	public static BiMap<String,Integer> constructGraph(String fileName) throws FileNotFoundException, IOException{
		BiMap<String,Integer> vertextId = HashBiMap.create();
		int idx = -1;
		List<String> numFromTos = new ArrayList<>();
		try(BufferedReader br = new BufferedReader(new FileReader(new File(fileName)))){
			String line;
            while ((line = br.readLine()) != null) {
            	String[] tokens = line.split(delim);
            	String from = tokens[0];
            	String to = tokens[1];
            	if( !vertextId.containsKey(from) )vertextId.put(from, ++idx);
            	if( !vertextId.containsKey(to) )vertextId.put(to, ++idx);
            	String numFromTo = vertextId.get(from) + delim + vertextId.get(to);
            	numFromTos.add(numFromTo);
            }
		}
		//
		try(PrintWriter pw = new PrintWriter("graphNum.txt")){
			for( String fromto : numFromTos )pw.println(fromto);
		}
		//
		return vertextId;
	}
	
	public static void main(String[] args) throws IOException {
		BiMap<String,Integer> bimapNameId = constructGraph("graph.txt");
		//
		Graph<String, String> graph = GraphLoader.loadUndirectedGraphEdgeListFile("graphNum.txt",bimapNameId.size(),delim);
		DeepWalk<String,String> dw = new DeepWalk.Builder<String,String>()
												.learningRate(0.01)
												.vectorSize(64)
												.windowSize(4)
												.build();
		dw.fit(graph, 3);
		//
		int[] num = dw.verticesNearest(bimapNameId.get("姚明"), 3);
		for( int n : num ){
			System.out.printf(format, bimapNameId.inverse().get(n), dw.similarity(1, n));
		}
	}

}
