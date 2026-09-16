outdoor_terrain_classification_database

A database of accelerometer, gyroscope, and magnetometer measurements for outdoor terrain classification. The database contains measurements for six outdoor terrain types in .csv format. The .zip file also contains a guide for both filenames and data structure. The detalis of the measurement setup can be found in the following open acces article:

Sarcevic, P.; Csík, D.; Pesti, R.; Stančin, S.; Tomažič, S.; Tadic, V.; Rodriguez-Resendiz, J.; Sárosi, J.; Odry, A. Online Outdoor Terrain Classification Algorithm for Wheeled Mobile Robots Equipped with Inertial and Magnetic Sensors. Electronics 2023, 12, x. https://doi.org/10.3390/xxxxx

The same article should be given as a reference when using the database.


///////////////////////
//GUIDE FOR FILENAMES//
///////////////////////

FORMAT:
"SEN_x_y_z.csv"


DESCRIPTION:

"SEN" - sensor type
Possible codes:
ACC - accelerometer
GYR - gyroscope
MAG - magnetometer

"x" - class code
Possible values:
1 - concrete
2 - grass
3 - pebbles
4 - sand
5 - paving stone
6 - synthetic running track

"y" - speed code
Possible values:
2 - 86% PWM
3 - 100% PWM

"z" - session code
Possible values:
1-7


//////////////////
//GUIDE FOR DATA//
//////////////////
1. column: X axis
2. column: Y axis
3. column: Z axis