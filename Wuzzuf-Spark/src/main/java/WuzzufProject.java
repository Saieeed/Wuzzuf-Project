// =====================================================================================================================
// The Project Libraries Used :-
// ------------------------------
import org.apache.spark.sql.catalyst.expressions.Sequence;
import org.apache.spark.sql.expressions.Window;
import scala.Tuple2;
import javax.swing.*;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.plans.*;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import static org.apache.spark.sql.functions.*;
// =====================================================================================================================


// =====================================================================================================================
public class WuzzufProject
{
    public static void main(String[] args)
    {
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            WuzzufProject WP = new WuzzufProject();

            // =========================================================================================================
            // Set Outputs Limit To Print According to It :-
            // ----------------------------------------------
            int Print_Limit = 10;

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            System.out.println("Initialize Spark Session");
            final SparkSession sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            DataFrameReader dataFrameReader = sparkSession.read().option("header", true);
            Dataset<Row> csvDataFrame = dataFrameReader.csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Print Part of The Dataset, Schema And Summary :-
            // -------------------------------------------------
            System.out.println("Wuzzuf DataSet Schema and Summary");
            csvDataFrame.show();
            csvDataFrame.printSchema();
            csvDataFrame.describe().show();

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
            System.out.println("Filtered Wuzzuf DataSet Summary");
            csvDataFrame = csvDataFrame
                    .filter(col("Title").notEqual("")).filter(col("Title").isNotNull())
                    .filter(col("Company").notEqual("")).filter(col("Company").isNotNull())
                    .filter(col("Location").notEqual("")).filter(col("Location").isNotNull())
                    .filter(col("Type").notEqual("")).filter(col("Type").isNotNull())
                    .filter(col("Level").notEqual("")).filter(col("Level").isNotNull())
                    .filter(col("YearsExp").notEqual("")).filter(col("YearsExp").isNotNull())
                    .filter(col("Country").notEqual("")).filter(col("Country").isNotNull())
                    .filter(col("Skills").notEqual("")).filter(col("Skills").isNotNull())
                    .distinct();
            csvDataFrame.describe().show();

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Jobs By Company :-
            // ------------------
            System.out.println("Jobs Ordered By Company :-");
            RelationalGroupedDataset JobsByCompany = csvDataFrame.groupBy("Company");
            Dataset <Row> CompanyJobs = JobsByCompany.count().sort(desc("count"));
            CompanyJobs.show(false);

            // =========================================================================================================
            // Plotting Jobs Per Company :-
            // -----------------------------
            List <String> CompanyName = CompanyJobs.select("Company").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> CompanyNumber = CompanyJobs.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.PieChart_Data(CompanyName,CompanyNumber,"Jobs Per Company");
            WP.BarChart_Data(CompanyName,CompanyNumber,"Jobs Per Company2");

            // =========================================================================================================
            // Plotting Jobs Per Company After Removing Confidential Companies :-
            // ------------------------------------------------------------------
            List <String> CompanyName2 = CompanyJobs.filter(col("Company").notEqual("Confidential"))
                                                    .select("Company").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> CompanyNumber2 = CompanyJobs.filter(col("Company").notEqual("Confidential"))
                                                    .select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.PieChart_Data(CompanyName2,CompanyNumber2,"Jobs Per Company Without Confidential");
            WP.BarChart_Data(CompanyName2,CompanyNumber2,"Jobs Per Company Without Confidential2");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Most Jobs Title :-
            // -------------------
            System.out.println("Jobs Ordered By Title :-");
            RelationalGroupedDataset JobsByTitle = csvDataFrame.groupBy("Title");
            Dataset <Row> TitleJob = JobsByTitle.count().sort(desc("count"));
            TitleJob.show(false);

            // =========================================================================================================
            // Plotting Jobs Per Title :-
            // ---------------------------
            List <String> TitleName = TitleJob.select("Title").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> TitleNumber = TitleJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(TitleName,TitleNumber,"Jobs Per Title");

            // =========================================================================================================
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Most Jobs Location :-
            // ----------------------
            System.out.println("Jobs Ordered By Location :-");
            RelationalGroupedDataset JobsByArea = csvDataFrame.groupBy("Location");
            Dataset <Row> AreaJob = JobsByArea.count().sort(desc("count"));
            AreaJob.show(false);

            // =========================================================================================================
            // Plotting Jobs Per Location :-
            // ------------------------------
            List <String> AreaName = AreaJob.select("Location").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> AreaNumber = AreaJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(AreaName,AreaNumber,"Jobs Per Location");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Most Jobs Skills :-
            // --------------------
            System.out.print("Jobs Ordered By Skills :-");
            Dataset <Row> SkillsJob = csvDataFrame.select("Skills")
                    .flatMap(row -> Arrays.asList(row.getString(0).split(",")).iterator(),Encoders.STRING())
                    .filter(s -> !s.isEmpty())
                    .map(word -> new Tuple2<>(word.toLowerCase(),1L),Encoders.tuple(Encoders.STRING(),Encoders.LONG()))
                    .toDF("Skills","count")
                    .groupBy("Skills")
                    .sum("count")
                    .orderBy(new Column("sum(count)").desc()).withColumnRenamed("sum(count)","count");
            SkillsJob.show(false);

            // =========================================================================================================
            // Plotting Jobs Per Skills :-
            // ----------------------------
            List <String> SkillsName = SkillsJob.select("Skills").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> SkillsNumber = SkillsJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(SkillsName,SkillsNumber,"Jobs Per Skills");
            // =========================================================================================================

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////// Bonus ///////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // =========================================================================================================
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Numbers of Years of Experience :-
            // ---------------------------------
            System.out.print("Number of Years Experience :-");
            Dataset<Row> YearsDF = csvDataFrame.withColumn("YearsExp", split(col("YearsExp"), "-|\\+| ").getItem(0))
                    .withColumn("YearsExp", when(col("YearsExp").equalTo("null"), "0").otherwise(col("YearsExp")))
                    .withColumn("YearsExp", col("YearsExp").cast("integer"));
            YearsDF.printSchema();
            RelationalGroupedDataset YearsExp = YearsDF.groupBy("YearsExp");
            Dataset<Row> Years_Exp = YearsExp.count().sort(asc("YearsExp"));
            Years_Exp.show(false);

            // =========================================================================================================
            // Plotting Jobs Per Years of Experience :-
            // -----------------------------------------
            List<String> YearsName = Years_Exp.select("YearsExp").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List<Long> YearsNumber = Years_Exp.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(YearsName, YearsNumber, "Jobs Per Years of Experience");

            // =========================================================================================================
            // Getting The List of Years of Experiences :-
            // -------------------------------------------
            List<String> ExpName = Years_Exp.select("YearsExp").as(Encoders.STRING()).collectAsList();

            // =========================================================================================================
            // Splitting Data To Lists According To Years of Experiences :-
            // -------------------------------------------------------------
            Dataset<Row>[] YearsExpList = new Dataset[ExpName.size()];
            for (int i = 0; i < ExpName.size(); i++)
            {
                Dataset<Row> current = YearsDF.filter(col("YearsExp").equalTo(ExpName.get(i)));
                YearsExpList[i] = current;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Apply Machine Learning For Job Titles And Companies :-
            // -------------------------------------------------------

            // A) Using Linear Regression :-
            // ------------------------------
            System.out.println("Predicting Job Title With Respect to Years of Experience Using Linear Regression");
            WP.ML_Prediction_Title(YearsDF);

            System.out.println("Predicting Company With Respect to Years of Experience Using Linear Regression");
            WP.ML_Prediction_Company(YearsDF);

            // B) Using KMeans :-
            // -------------------
            System.out.println("Predicting Job Title With Respect to Years of Experience Using KMeans");
            WP.KMeans_Prediction_Title(YearsDF, ExpName.size());

            System.out.println("Predicting Company With Respect to Years of Experience Using KMeans");
            WP.KMeans_Prediction_Company(YearsDF, ExpName.size());

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            System.out.println("Terminating Spark Session");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void BarChart_Data(List <String> Name , List <Long> Number, String SeriesTitle)
    {
        try
        {
            // =========================================================================================================
            // XChart BarChart Plotting :-
            // ----------------------------
            CategoryChart chart = new CategoryChartBuilder().width(1024).height(800).title(SeriesTitle)
                                                            .xAxisTitle ("Name").yAxisTitle ("Number").build ();
            chart.getStyler ().setLegendPosition (Styler.LegendPosition.InsideNW).setHasAnnotations (true);
            chart.getStyler ().setStacked(true);
            chart.addSeries (SeriesTitle, Name, Number);

            // =========================================================================================================
            // Exporting BarChart As JPG To Be Streamed :-
            // --------------------------------------------
            new SwingWrapper(chart).displayChart ().setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            BitmapEncoder.saveBitmap(chart, "./src/main/resources/Graphs/" + SeriesTitle, BitmapEncoder.BitmapFormat.JPG);
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void PieChart_Data(List <String> Name , List <Long> Number, String SeriesTitle)
    {
        try
        {
            // =========================================================================================================
            // XChart PieChart Plotting :-
            // ----------------------------
            PieChart chart = new PieChartBuilder().width(1024).height(800).title(SeriesTitle).build ();
            for (int i = 0; i < Name.size();i++ )
            {
                chart.addSeries (Name.get(i), Number.get(i));
            }

            // =========================================================================================================
            // Exporting PieChart As JPG To Be Streamed :-
            // --------------------------------------------
            new SwingWrapper (chart).displayChart ().setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            BitmapEncoder.saveBitmap(chart, "./src/main/resources/Graphs/" + SeriesTitle, BitmapEncoder.BitmapFormat.JPG);
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void ML_Prediction_Title(Dataset<Row> DataFrame)
    {
        try
        {
            // =========================================================================================================
            // Add Title ID Column :-
            // -----------------------
            DataFrame = DataFrame.withColumn("TitleID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double split[] = {0.8, 0.2};
            Dataset<Row> DFArray[] = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            System.out.println("Training Data Set Size is " + DFTrain.count());
            System.out.println("Test Data Set Size is " + DFTest.count());

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // ---------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String inputColumns[] = {"YearsExp"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform The Train Dataset Using VectorAssembler.transform :-
            // --------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            DFTrainTransform.select("TitleID","Title","YearsExp", "features").show();

            // =========================================================================================================
            // Create A LinearRegression Estimator and Set The Feature Column And The Label Column :-
            // ---------------------------------------------------------------------------------------
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setFeaturesCol("features");
            linearRegression.setLabelCol("TitleID");

            // =========================================================================================================
            // Fit The Linear Regression Model :-
            // -----------------------------------
            LinearRegressionModel linearRegressionModel = linearRegression.fit(DFTrainTransform);
            double coefficient = Math.round(linearRegressionModel.coefficients().toArray()[0]);
            double intercept = Math.round(linearRegressionModel.intercept());
            System.out.println("The formula for the linear regression line is price = " + coefficient + " * YearsExp + " + intercept);

            // =========================================================================================================
            // Printing The Prediction :-
            // ---------------------------
            DFTest = vectorAssembler.transform(DFTest.na().drop());
            final Dataset<Row> predictions = linearRegressionModel.transform(DFTest);
            predictions.show();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void ML_Prediction_Company(Dataset<Row> DataFrame)
    {
        try
        {
            // =========================================================================================================
            // Add Company ID Column :-
            // -------------------------
            DataFrame = DataFrame.withColumn("CompanyID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double split[] = {0.8, 0.2};
            Dataset<Row> DFArray[] = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            System.out.println("Training Data Set Size is " + DFTrain.count());
            System.out.println("Test Data Set Size is " + DFTest.count());

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // ---------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String inputColumns[] = {"YearsExp"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // --------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            DFTrainTransform.select("CompanyID","Company","YearsExp", "features").show();

            // =========================================================================================================
            // Create a LinearRegression Estimator And Set The Feature Column And The Label Column :-
            // ---------------------------------------------------------------------------------------
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.setFeaturesCol("features");
            linearRegression.setLabelCol("CompanyID");

            // =========================================================================================================
            // Fit The Linear Regression Model :-
            // -----------------------------------
            LinearRegressionModel linearRegressionModel = linearRegression.fit(DFTrainTransform);
            double coefficient = Math.round(linearRegressionModel.coefficients().toArray()[0]);
            double intercept = Math.round(linearRegressionModel.intercept());
            System.out.println("The formula for the linear regression line is price = " + coefficient + " * YearsExp + " + intercept);

            // =========================================================================================================
            // Printing The Prediction :-
            // ---------------------------
            DFTest = vectorAssembler.transform(DFTest.na().drop());
            final Dataset<Row> predictions = linearRegressionModel.transform(DFTest);
            predictions.show();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void KMeans_Prediction_Title(Dataset<Row> DataFrame,int k)
    {
        try
        {
            // =========================================================================================================
            // Add Title ID Column :-
            // -----------------------
            DataFrame = DataFrame.withColumn("TitleID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double split[] = {0.8, 0.2};
            Dataset<Row> DFArray[] = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            System.out.println("Training Data Set Size is " + DFTrain.count());
            System.out.println("Test Data Set Size is " + DFTest.count());

            // =========================================================================================================
            // Create the Vector Assembler That Will Contain The Feature Columns :-
            // --------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String inputColumns[] = {"TitleID"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // ---------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            DFTrainTransform.select("TitleID","Title","YearsExp", "features").show();

            // =========================================================================================================
            // Trains The KMeans Model :-
            // --------------------------
            KMeans kmeans = new KMeans().setK(k).setSeed(1L);
            kmeans.setFeaturesCol("features");
            kmeans.setPredictionCol("Predicted");
            KMeansModel model = kmeans.fit(DFTrainTransform);

            // =========================================================================================================
            // Printing The Predictions :-
            // ----------------------------
            Dataset<Row> predictions = model.transform(DFTrainTransform);
            predictions.show();

            // =========================================================================================================
            // Printing KMeans Clustering Centers :-
            // --------------------------------------
            System.out.println("KMeans Clusters Centers = ");
            for (Object o : model.clusterCenters())
            {
                System.out.print(o + " , ");
            }
            System.out.println();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public void KMeans_Prediction_Company(Dataset<Row> DataFrame,int k)
    {
        try
        {
            // =========================================================================================================
            // Add Company ID Column :-
            // -------------------------
            DataFrame = DataFrame.withColumn("CompanyID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double split[] = {0.8, 0.2};
            Dataset<Row> DFArray[] = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            System.out.println("Training Data Set Size is " + DFTrain.count());
            System.out.println("Test Data Set Size is " + DFTest.count());

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // --------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String inputColumns[] = {"CompanyID"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // ---------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            DFTrainTransform.select("CompanyID","Company","YearsExp", "features").show();

            // =========================================================================================================
            // Trains The KMeans Model :-
            // --------------------------
            KMeans kmeans = new KMeans().setK(k).setSeed(1L);
            kmeans.setFeaturesCol("features");
            kmeans.setPredictionCol("Predicted");
            KMeansModel model = kmeans.fit(DFTrainTransform);

            // =========================================================================================================
            // Printing The Predictions :-
            // ----------------------------
            Dataset<Row> predictions = model.transform(DFTrainTransform);
            predictions.show();

            // =========================================================================================================
            // Printing KMeans Clustering Centers :-
            // --------------------------------------
            System.out.println("KMeans Clusters Centers = ");
            for (Object o : model.clusterCenters())
            {
                System.out.print(o + " , ");
            }
            System.out.println();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }
}