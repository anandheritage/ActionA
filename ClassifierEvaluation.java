import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ClassifierEvaluation
{
	public static void main(String args[])throws Exception
	{
			BufferedReader breader=new BufferedReader(new FileReader("G:/Program Files/Weka-3-6/data/diabetes.arff"));
			Instances train=new Instances(breader);
			train.setClassIndex(train.numAttributes()-1);
			breader.close();
			NaiveBayes nB=new NaiveBayes();
			nB.buildClassifier(train);
			Evaluation eval=new Evaluation(train);
			eval.crossValidateModel(nB, train, 10, new Random(1));
			System.out.print(eval.toSummaryString("Results for J48  ----\n",true));
			J48 tree=new J48();
			tree.buildClassifier(train);
			eval=new Evaluation(train);
			eval.crossValidateModel(tree, train, 10, new Random(1));
			System.out.print(eval.toSummaryString("Results  for NB ----\n",true));
	}

}
