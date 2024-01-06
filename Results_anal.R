# Assuming index is a variable with the index value
suffix <- "GT_scores_and_diameters.csv"

# Define the path
path_base <- "C://Users//Chris//OneDrive//2023//BME_M2//Image_Processing//Project//Results"

# Create a list to store dataframes
donnees_list <- list()

# Loop through index values from 1 to 7
for (i in 1:7) {
  # Format the path using paste
  path <- paste(path_base, i, suffix, sep = "/")
  
  # Read the CSV file into a dataframe
  donnees <- read.csv(path)
  
  # Assign the dataframe to a list element
  donnees_list[[paste0("donnees_", i)]] <- donnees
}

# mean dice scores and bar plot #####

# Now, donnees_list contains dataframes donnees_1, donnees_2, ..., donnees_7
Means_Asc = c()
Means_Desc = c()
for (i in 1:length(donnees_list)){
  print(paste0("Image:",i))
  print(mean(na.omit(donnees_list[[i]]$Descending_Scores)))
  Means_Desc = c(Means_Desc, mean(na.omit(donnees_list[[i]]$Descending_Diameters)))
  print(mean(na.omit(donnees_list[[i]]$Ascending_Scores)))
  Means_Asc = c(Means_Asc, mean(na.omit(donnees_list[[i]]$Ascending_Diameters)))
}

mean(Means_Asc)
sd(Means_Asc)

# Combine means into a data frame
means_data <- data.frame(
  Sample = 1:7,
  Ascending = Means_Asc,
  Descending = Means_Desc
)

z = c(100,100,100,70,60, 86, 86)

t1 = c(88,95,90,90,91, 90.8, 90.8)

t2 = c(95,97,95,95,97,95,97)

# Reshape the data for ggplot2
library(reshape2)
means_data_long <- melt(means_data, id.vars = "Sample")

# Plot using ggplot2
library(ggplot2)
ggplot(means_data_long, aes(x = factor(Sample), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Sample", y = "Mean Dice Score", title = "Mean Dice Scores on Training Set") +
  scale_fill_manual(values = c("hotpink", "darkgrey")) +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 1))



#Plotting scores vs slice##### 

# Create a larger plot with the first set of scores
plot(donnees_list[[1]]$Slice, donnees_list[[1]]$Descending_Scores, type = "l", col = "blue", lty = 1, xlab = "Slice", ylab = "Dice score", main = "Training Set Dice Scores for Descending Aorta", ylim = c(0, 1), xlim = c(0, 400), cex.main = 1.2)

# Overlay the rest of the scores
for (i in 2:7) {
  lines(donnees_list[[i]]$Slice, donnees_list[[i]]$Descending_Scores, col = rainbow(7)[i], lty = i)
}

# Add legend further outside the plot
par(xpd = TRUE)  # Allow drawing outside the plot area
legend("topright", inset = c(0, 0), legend = c("Scan 1", paste("Scan", 2:7)), 
       col = c("blue", rainbow(6)), lty = c(1, 2:7), bty = "n", cex = 0.8)
par(xpd = FALSE)  # Reset to default behavior


# Create a larger plot with the first set of scores
plot(donnees_list[[1]]$Slice, donnees_list[[1]]$Ascending_Scores, type = "l", col = "blue", lty = 1, xlab = "Slice", ylab = "Dice score", main = "Training Set Dice Scores for Ascending Aorta", ylim = c(0, 1), xlim = c(0, 150), cex.main = 1.2)

# Overlay the rest of the scores
for (i in 2:7) {
  lines(donnees_list[[i]]$Slice, donnees_list[[i]]$Ascending_Scores, col = rainbow(7)[i], lty = i)
}

# Add legend further outside the plot
par(xpd = TRUE)  # Allow drawing outside the plot area
legend("topright", inset = c(0, 0), legend = c("Scan 1", paste("Scan", 2:7)), 
       col = c("blue", rainbow(6)), lty = c(1, 2:7), bty = "n", cex = 0.8)
par(xpd = FALSE)  # Reset to default behavior




gtAAo = c()
gtdAo = c()
# Create a larger plot with the first set of scores
plot(donnees_list[[1]]$Slice, donnees_list[[1]]$Ascending_Diameters, type = "l", col = "blue", lty = 1, xlab = "Slice", ylab = "Diameter", main = "Training Set Diameters for Ascending Aorta", ylim = c(1,7),xlim = c(0, 400), cex.main = 1.2)

# Overlay the rest of the scores
for (i in 1:7) {
  lines(donnees_list[[i]]$Slice, donnees_list[[i]]$Ascending_Diameters, col = rainbow(7)[i], lty = i)
}

# Add legend further outside the plot
par(xpd = TRUE)  # Allow drawing outside the plot area
legend("topright", inset = c(0, 0), legend = c("Scan 1", paste("Scan", 2:7)), 
       col = c("blue", rainbow(6)), lty = c(1, 7), bty = "n", cex = 0.8)
par(xpd = FALSE)  # Reset to default behavior


# Create a larger plot with the first set of scores
plot(donnees_list[[1]]$Slice, donnees_list[[1]]$Ascending_Diameters, type = "l", col = "blue", lty = 1, xlab = "Slice", ylab = "Diameters", main = "Training Set Diameters for Ascending Aorta", xlim = c(0, 150), cex.main = 1.2)

# Overlay the rest of the scores
for (i in c(2:7)) {
  lines(donnees_list[[i]]$Slice, donnees_list[[i]]$Ascending_Diameters, col = rainbow(7)[i], lty = i)
}

# Add legend further outside the plot
par(xpd = TRUE)  # Allow drawing outside the plot area
legend("topright", inset = c(0, 0), legend = c("Scan 1", paste("Scan", c(2:7))), 
       col = c("blue", rainbow(6)), lty = c(1, 4), bty = "n", cex = 0.8)
par(xpd = FALSE)  # Reset to default behavior