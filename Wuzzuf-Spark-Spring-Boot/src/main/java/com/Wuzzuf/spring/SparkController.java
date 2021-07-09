// =====================================================================================================================
package com.Wuzzuf.spring;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import scala.Tuple2;
import javax.swing.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.ui.Model;
import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;
// =====================================================================================================================

// =====================================================================================================================
@RequestMapping("Wuzzuf")
@Controller
public class SparkController
{

    @Autowired
    private SparkSession sparkSession;

    @GetMapping("")
    public String index(Model model)
    {

        return "index";
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // =================================================================================================================
    // Wuzzuf Main Look :-
    // --------------------

    @GetMapping(value = "/body_bg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource>body_bg() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\body_bg.png")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    @GetMapping(value = "/logo", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource>logo() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\logo.ico")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    @GetMapping(value = "/box", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource>box() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\box_img.png")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // =================================================================================================================
    // Wuzzuf Main Requests :-
    // ------------------------

    @RequestMapping("ReadCSV")
    public ResponseEntity<String> Wuzzuf_ReadCSV()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html ="";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        // =============================================================================================================
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");
            html = html + String.format("<h1>%s</h1>", "Running Apache Spark on/with support of Spring boot") +
                   String.format("<h2>%s</h2>", "Spark version = " + sparkSession.sparkContext().version()) +
                    String.format("<h3>%s</h3>", "Read csv..") +
                    String.format("<h4>Total records %d</h4>", csvDataFrame.count()) +
                    String.format("<h5>Schema <br/> %s</h5> ", csvDataFrame.schema().treeString()) +
                   String.format("<h5/> Sample data <br/></h5>" + csvDataFrame.showString(20, 20, true)) +
                    String.format("<h5> Describe <br/></h5>" + csvDataFrame.describe().showString(20,20,true));

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
            html = html + String.format("<h5>Filtered Wuzzuf DataSet Summary</h5>");
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
            html = html + String.format(csvDataFrame.describe().showString(20,20,true));

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    @RequestMapping("JobsByCompany")
    public ResponseEntity<String> Wuzzuf_JobsByCompany()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        // =============================================================================================================
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Jobs By Company :-
            // ------------------
            html = html + String.format("<h5>Jobs Ordered By Company :-</h5>");
            RelationalGroupedDataset JobsByCompany = csvDataFrame.groupBy("Company");
            Dataset<Row> CompanyJobs = JobsByCompany.count().sort(desc("count"));
            html = html + String.format(CompanyJobs.showString(20, 20, true));

            // =========================================================================================================
            // Plotting Jobs Per Company :-
            // -----------------------------
            List<String> CompanyName = CompanyJobs.select("Company").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List<Long> CompanyNumber = CompanyJobs.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
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
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerCompanyImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource>JobsPerCompanyImg() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Company.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerCompanyWithoutConfImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource> JobsPerCompanyWithoutConfImg() throws IOException
    {
        final ByteArrayResource inputStream =
                new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Company Without Confidential.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @RequestMapping("JobsByTitle")
    public ResponseEntity<String> Wuzzuf_JobsByTitle()
    {
        // =========================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =========================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        // =============================================================================================================
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Most Jobs Title :-
            // -------------------
            html = html + String.format("<h5>Jobs Ordered By Title :-</h5>");
            RelationalGroupedDataset JobsByTitle = csvDataFrame.groupBy("Title");
            Dataset <Row> TitleJob = JobsByTitle.count().sort(desc("count"));
            html = html + String.format(TitleJob.showString(20,20,true));

            // =========================================================================================================
            // Plotting Jobs Per Title :-
            // ---------------------------
            List <String> TitleName = TitleJob.select("Title").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> TitleNumber = TitleJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(TitleName,TitleNumber,"Jobs Per Title");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerTitleImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource> JobsPerTitleImg() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Title.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @RequestMapping("JobsByLocations")
    public ResponseEntity<String> Wuzzuf_JobsByLocations()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        // =============================================================================================================
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            // =========================================================================================================
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Most Jobs Location :-
            // ----------------------
            html = html + String.format("<h5>Jobs Ordered By Location :-</h5>");
            RelationalGroupedDataset JobsByArea = csvDataFrame.groupBy("Location");
            Dataset <Row> AreaJob = JobsByArea.count().sort(desc("count"));
            html = html + String.format(AreaJob.showString(20,20,true));

            // =========================================================================================================
            // Plotting Jobs Per Location :-
            // ------------------------------
            List <String> AreaName = AreaJob.select("Location").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> AreaNumber = AreaJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(AreaName,AreaNumber,"Jobs Per Location");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerLocationImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource> JobsPerLocationImg() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Location.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @RequestMapping("JobsBySkills")
    public ResponseEntity<String> Wuzzuf_JobsBySkills()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        // =============================================================================================================
        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Most Jobs Skills :-
            // --------------------
            html = html + String.format("<h5>Jobs Ordered By Skills :-</h5>");
            Dataset <Row> SkillsJob = csvDataFrame.select("Skills")
                    .flatMap(row -> Arrays.asList(row.getString(0).split(",")).iterator(), Encoders.STRING())
                    .filter(s -> !s.isEmpty())
                    .map(word -> new Tuple2<>(word.toLowerCase(),1L),Encoders.tuple(Encoders.STRING(),Encoders.LONG()))
                    .toDF("Skills","count")
                    .groupBy("Skills")
                    .sum("count")
                    .orderBy(new Column("sum(count)").desc()).withColumnRenamed("sum(count)","count");
            html = html + String.format(SkillsJob.showString(20,20,true));

            // =========================================================================================================
            // Plotting Jobs Per Skills :-
            // ----------------------------
            List <String> SkillsName = SkillsJob.select("Skills").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List <Long> SkillsNumber = SkillsJob.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(SkillsName,SkillsNumber,"Jobs Per Skills");

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerSkillsImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource> JobsPerSkillsImg() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Skills.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @RequestMapping("JobsByYears")
    public ResponseEntity<String> Wuzzuf_JobsByYears()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            // =========================================================================================================
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Numbers of Years of Experience :-
            // ---------------------------------
            html = html + String.format("<h5>Number of Years Experience :-</h5>");
            Dataset<Row> YearsDF = csvDataFrame.withColumn("YearsExp", split(col("YearsExp"), "-|\\+| ").getItem(0))
                    .withColumn("YearsExp", when(col("YearsExp").equalTo("null"), "0").otherwise(col("YearsExp")))
                    .withColumn("YearsExp", col("YearsExp").cast("integer"));
            html = html + String.format(YearsDF.schema().treeString()) + "</br>";
            RelationalGroupedDataset YearsExp = YearsDF.groupBy("YearsExp");
            Dataset<Row> Years_Exp = YearsExp.count().sort(asc("YearsExp"));
            html = html + String.format(Years_Exp.showString(20,20,true));

            // =========================================================================================================
            // Plotting Jobs Per Years of Experience :-
            // -----------------------------------------
            List<String> YearsName = Years_Exp.select("YearsExp").limit(Print_Limit).as(Encoders.STRING()).collectAsList();
            List<Long> YearsNumber = Years_Exp.select("count").limit(Print_Limit).as(Encoders.LONG()).collectAsList();
            WP.BarChart_Data(YearsName, YearsNumber, "Jobs Per Years of Experience");

            // =========================================================================================================
            // Getting The List of Years of Experiences :-
            // -------------------------------------------
            List<String> ExpName = Years_Exp.select("YearsExp").sort("YearsExp").as(Encoders.STRING()).collectAsList();

            // =========================================================================================================
            // Splitting Data To Lists According To Years of Experiences :-
            // -------------------------------------------------------------
            html = html + String.format("</br></br>Printing Each Year of Experiences Alone</br></br>");
            Dataset<Row>[] YearsExpList = new Dataset[ExpName.size()];
            for (int i = 0; i < ExpName.size(); i++)
            {
                html = html + "</br></br> - "+ ExpName.get(i) + " Years of Experiences</br></br>";
                Dataset<Row> current = YearsDF.filter(col("YearsExp").equalTo(ExpName.get(i)));
                YearsExpList[i] = current;
                html = html + String.format(current.showString(20,20,true)) + "</br></br>";
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @GetMapping(value = "/JobsPerYearsofExperienceImg", produces = MediaType.IMAGE_JPEG_VALUE)
    public ResponseEntity<Resource> JobsPerYearsofExperience() throws IOException
    {
        final ByteArrayResource inputStream = new ByteArrayResource(Files.readAllBytes(Paths.get("src\\main\\webapp\\images\\Jobs Per Years of Experience.jpg")));
        return ResponseEntity.status(HttpStatus.OK).contentLength(inputStream.contentLength()).body(inputStream);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @RequestMapping("JobsPrediction")
    public ResponseEntity<String> Wuzzuf_MachineLearning()
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        // =============================================================================================================
        // Set Outputs Limit To Print According to It :-
        // ----------------------------------------------
        int Print_Limit = 10;

        try
        {
            // =========================================================================================================
            // Set Logs To Print Errors Only :-
            // ---------------------------------
            Logger.getLogger("org").setLevel(Level.ERROR);

            // =========================================================================================================
            // Create An Object From The Main Class :-
            // ----------------------------------------
            SparkController WP = new SparkController();

            // =========================================================================================================
            // Create Spark Session To Create Connection To Spark :-
            // -----------------------------------------------------
            html = html + String.format("Initialize Spark Session </br>");
            sparkSession = SparkSession.builder().appName("Wuzzuf Dataset").master("local[8]").getOrCreate();

            // =========================================================================================================
            // Get DataFrameReader using SparkSession :-
            // ------------------------------------------
            Dataset<Row> csvDataFrame = sparkSession.read().option("header", "true").csv("src/main/resources/Wuzzuf_Jobs.csv");

            // =========================================================================================================
            // Remove Empty and Duplicated Values Then Print Summary :-
            // ---------------------------------------------------------
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

            // =========================================================================================================
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Numbers of Years of Experience :-
            // ---------------------------------
            html = html + String.format("<h5>Number of Years Experience :-</h5>");
            Dataset<Row> YearsDF = csvDataFrame.withColumn("YearsExp", split(col("YearsExp"), "-|\\+| ").getItem(0))
                    .withColumn("YearsExp", when(col("YearsExp").equalTo("null"), "0").otherwise(col("YearsExp")))
                    .withColumn("YearsExp", col("YearsExp").cast("integer"));
            RelationalGroupedDataset YearsExp = YearsDF.groupBy("YearsExp");
            Dataset<Row> Years_Exp = YearsExp.count().sort(asc("YearsExp"));

            // =========================================================================================================
            // Getting The List of Years of Experiences :-
            // -------------------------------------------
            List<String> ExpName = Years_Exp.select("YearsExp").as(Encoders.STRING()).collectAsList();

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Apply Machine Learning For Job Titles And Companies :-
            // -------------------------------------------------------

            // A) Using Linear Regression :-
            // ------------------------------
            html = html + String.format("<h5>Predicting Job Title With Respect to Years of Experience Using Linear Regression</h5>");
            html = html + WP.ML_Prediction_Title(YearsDF);

            html = html + String.format("<h5>Predicting Company With Respect to Years of Experience Using Linear Regression</h5>");
            html = html + WP.ML_Prediction_Company(YearsDF);

            // B) Using KMeans :-
            // -------------------
            html = html + String.format("<h5>Predicting Job Title With Respect to Years of Experience Using KMeans</h5>");
            html = html + WP.KMeans_Prediction_Title(YearsDF, ExpName.size());

            html = html + String.format("<h5>Predicting Job Title With Respect to Years of Experience Using KMeans</h5>");
            html = html + WP.KMeans_Prediction_Company(YearsDF, ExpName.size());

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // =========================================================================================================
            // Terminating Spark Session :-
            // -----------------------------
            html = html + String.format("<h5>Terminating Spark Session</h5>");
            sparkSession.stop();
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }
        return ResponseEntity.ok(html);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////// Other Functions Used /////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public void BarChart_Data(List<String> Name , List <Long> Number, String SeriesTitle)
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
            BitmapEncoder.saveBitmap(chart, "src\\main\\webapp\\images\\" + SeriesTitle, BitmapEncoder.BitmapFormat.JPG);
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
            BitmapEncoder.saveBitmap(chart, "src\\main\\webapp\\images\\" + SeriesTitle, BitmapEncoder.BitmapFormat.JPG);
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
        }
    }

    public String ML_Prediction_Title(Dataset<Row> DataFrame)
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        try
        {
            // =========================================================================================================
            // Add Title ID Column :-
            // -----------------------
            DataFrame = DataFrame.withColumn("TitleID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double[] split = {0.8, 0.2};
            Dataset<Row>[] DFArray = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            html = html + "Training Data Set Size is " + DFTrain.count() + " </br>";
            html = html + "Test Data Set Size is " + DFTest.count() + " </br>";

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // ---------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String[] inputColumns = {"YearsExp"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform The Train Dataset Using VectorAssembler.transform :-
            // --------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            html = html + String.format(DFTrainTransform.select("TitleID","Title","YearsExp", "features")
                    .showString(20, 20, true));

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
            html = html + "The formula for the linear regression line is price = " + coefficient + " * YearsExp + " + intercept  + " </br>";

            // =========================================================================================================
            // Printing The Prediction :-
            // ---------------------------
            DFTest = vectorAssembler.transform(DFTest.na().drop());
            final Dataset<Row> predictions = linearRegressionModel.transform(DFTest);
            html = html + String.format(predictions.showString(20, 20, true));
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }

        // =============================================================================================================
        // Printing Output :-
        // -------------------
        return html;
    }

    public String ML_Prediction_Company(Dataset<Row> DataFrame)
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        try
        {
            // =========================================================================================================
            // Add Company ID Column :-
            // -------------------------
            DataFrame = DataFrame.withColumn("CompanyID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double[] split = {0.8, 0.2};
            Dataset<Row>[] DFArray = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            html = html + "Training Data Set Size is " + DFTrain.count() + " </br>";
            html = html + "Test Data Set Size is " + DFTest.count() + " </br>";

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // ---------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String[] inputColumns = {"YearsExp"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // --------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            html = html + String.format(DFTrainTransform.select("CompanyID","Company","YearsExp", "features")
                    .showString(20, 20, true));

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
            html = html + "The formula for the linear regression line is price = " + coefficient + " * YearsExp + " + intercept  + " </br>";

            // =========================================================================================================
            // Printing The Prediction :-
            // ---------------------------
            DFTest = vectorAssembler.transform(DFTest.na().drop());
            final Dataset<Row> predictions = linearRegressionModel.transform(DFTest);
            html = html + String.format(predictions.showString(20, 20, true));
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }

        // =============================================================================================================
        // Printing Output :-
        // -------------------
        return html;
    }

    public String KMeans_Prediction_Title(Dataset<Row> DataFrame,int k)
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        try
        {
            // =========================================================================================================
            // Add Title ID Column :-
            // -----------------------
            DataFrame = DataFrame.withColumn("TitleID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double[] split = {0.8, 0.2};
            Dataset<Row>[] DFArray = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            html = html + "Training Data Set Size is " + DFTrain.count() + " </br>";
            html = html + "Test Data Set Size is " + DFTest.count() + " </br>";

            // =========================================================================================================
            // Create the Vector Assembler That Will Contain The Feature Columns :-
            // --------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String[] inputColumns = {"TitleID"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // ---------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            html = html + String.format(DFTrainTransform.select("TitleID","Title","YearsExp", "features")
                    .showString(20, 20, true));

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
            html = html + String.format(predictions.showString(20, 20, true));

            // =========================================================================================================
            // Printing KMeans Clustering Centers :-
            // --------------------------------------
            html = html + "KMeans Clusters Centers = </br>";
            for (Object o : model.clusterCenters())
            {
                html = html + o.toString() + " , ";
            }
            html = html + "</br>";
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }

        // =============================================================================================================
        // Printing Output :-
        // -------------------
        return html;
    }

    public String KMeans_Prediction_Company(Dataset<Row> DataFrame,int k)
    {
        // =============================================================================================================
        // HTML Output Initialization :-
        // -----------------------------
        String html = "";

        try
        {
            // =========================================================================================================
            // Add Company ID Column :-
            // -------------------------
            DataFrame = DataFrame.withColumn("CompanyID",functions.monotonically_increasing_id());

            // =========================================================================================================
            // Randomly Split The Dataset To 80% Train Data And 20% Test Data :-
            // --------------------------------------------------------------------
            double[] split = {0.8, 0.2};
            Dataset<Row>[] DFArray = DataFrame.randomSplit(split, 42);
            Dataset<Row> DFTrain = DFArray[0];
            Dataset<Row> DFTest = DFArray[1];
            html = html + "Training Data Set Size is " + DFTrain.count() + " </br>";
            html = html + "Test Data Set Size is " + DFTest.count() + " </br>";

            // =========================================================================================================
            // Create The Vector Assembler That Will Contain The Feature Columns :-
            // --------------------------------------------------------------------
            VectorAssembler vectorAssembler = new VectorAssembler();
            String[] inputColumns = {"CompanyID"};
            vectorAssembler.setInputCols(inputColumns);
            vectorAssembler.setOutputCol("features");

            // =========================================================================================================
            // Transform the Train Dataset Using VectorAssembler.transform :-
            // ---------------------------------------------------------------
            Dataset<Row> DFTrainTransform = vectorAssembler.transform(DFTrain.na().drop());
            html = html + String.format(DFTrainTransform.select("TitleID","Title","YearsExp", "features")
                    .showString(20, 20, true));

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
            html = html + String.format(predictions.showString(20, 20, true));

            // =========================================================================================================
            // Printing KMeans Clustering Centers :-
            // --------------------------------------
            html = html + "KMeans Clusters Centers = </br>";
            for (Object o : model.clusterCenters())
            {
                html = html + o + " , ";
            }
            html = html + "</br>";
        }
        catch (Exception e)
        {
            System.out.println(e.getMessage());
            html = html + e.getMessage();
        }

        // =============================================================================================================
        // Printing Output :-
        // -------------------
        return html;
    }
}