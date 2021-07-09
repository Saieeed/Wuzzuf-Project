package com.Wuzzuf;
import java.util.Arrays;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.support.SpringBootServletInitializer;
import org.springframework.context.ApplicationContext;


@SpringBootApplication
public class Main extends SpringBootServletInitializer
{
    public static void main(String[] args)
    {
        Logger.getLogger("org").setLevel(Level.ERROR);
        ApplicationContext applicationContext = SpringApplication.run(Main.class, args);
        String[] beanNames = applicationContext.getBeanDefinitionNames();
        Arrays.sort(beanNames);
        for (String beanName : beanNames)
        {
            System.out.println(beanName);
        }
    }
}