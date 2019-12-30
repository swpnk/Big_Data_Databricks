// Databricks notebook source
// MAGIC %fs ls /FileStore/tables/retail_db/orders

// COMMAND ----------

val orders = spark.read.
schema("order_id INT, order_date TIMESTAMP, order_customer_id INT, order_status STRING").
csv("/FileStore/tables/retail_db/orders/")

// COMMAND ----------

val orderItems = spark.read.
schema("order_item_id INT, order_item_order_id INT, order_item_product_id INT, order_quantity INT, order_item_subtotal FLOAT, order_item_product_price FLOAT").
csv("/FileStore/tables/retail_db/order_items")

// COMMAND ----------

val products = spark.read.
schema("product_id INT, prodcut_category_id INT, product_name STRING, product_description STRING, product_price FLOAT, product_image STRING").
csv("/FileStore/tables/retail_db/products")

// COMMAND ----------

import org.apache.spark.sql.functions._



// COMMAND ----------

//Get Distinct Order Statuses
orders.select((col("order_status"))).distinct().show()


// COMMAND ----------

// Get OrderStatus Count 
val orderStatusCount = orders.groupBy("order_status").agg(count(lit(1)).alias("order_count"))

// COMMAND ----------

display(orderStatusCount)

// COMMAND ----------

val ordersCompleted = orders.filter("order_status IN ('COMPLETE','CLOSED')")

// COMMAND ----------

val joinResults = ordersCompleted.join(orderItems, $"order_id" === $"order_item_order_id").join(products, $"product_id" === $"order_item_order_id").
select("order_date","product_name","order_item_subtotal")

// COMMAND ----------

joinResults.show()


// COMMAND ----------

val prodcutRev = joinResults.groupBy("order_date","product_name").agg(round(sum("order_item_subtotal"),2).alias("product_revenue"))

// COMMAND ----------

prodcutRev.show(false)

// COMMAND ----------

val sortedrev = prodcutRev.orderBy($"product_revenue".desc)

// COMMAND ----------

sortedrev.show()

// COMMAND ----------

sortedrev.write.mode.("overwrite").csv("FileStore/tables/retail_db/daily_product_revenue")

// COMMAND ----------

val customers = spark.read.format("csv").schema("customer_id INT, customer_fname STRING, customer_lname STRING, customer_email STRING, customer_password STRING, customer_street STRING, customer_city STRING, customer_state STRING, customer_zipcode STRING").load("/FileStore/tables/retail_db/customers")

// COMMAND ----------



// COMMAND ----------

val customerconcat = customers.withColumn("customer_name", concat(col("customer_fname"),lit(" "),(col("customer_lname"))))

// COMMAND ----------

customerconcat.show(1)

// COMMAND ----------

val customerFiltered = customerconcat.select("customer_id","customer_name")

// COMMAND ----------

customerFiltered.show(1)

// COMMAND ----------

orders.show(1)

// COMMAND ----------

val ordersmonth = orders.withColumn("yearmonth", date_format($"order_date", "YYYY-MM"))

// COMMAND ----------

ordersmonth.show(1)

// COMMAND ----------

val ordersFiltered = ordersmonth.select("order_id","order_customer_id","yearmonth")

// COMMAND ----------

val joined = ordersFiltered.join(customerFiltered, $"order_customer_id" === $"customer_id").join(orderItems, $"order_item_id" === $"order_id").select("yearmonth", "customer_name","order_item_subtotal")

// COMMAND ----------

joined.show(1)

// COMMAND ----------

val revunuecustomermonth = joined.groupBy($"yearmonth",$"customer_name").agg(sum($"order_item_subtotal").alias("Revenue")).orderBy($"yearmonth".asc,$"Revenue".desc)

// COMMAND ----------

revunuecustomermonth.show(20)

// COMMAND ----------

