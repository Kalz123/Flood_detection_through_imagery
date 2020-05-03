# 5-3-20 Meet 1

## Project Prompt 12

Problem 12: Extracting Flood Depths from Imagery

Problem Statement: Floods cause damage to infrastructure and homes. The depth of flood waters is a good indicator of the severity of damage. Floods are incredibly difficult to model, and while model outputs are useful to emergency managers, it is crucial to know the actual depth. Social media and news outlets often present pictures of floods. 

**How can this imagery be used to estimate the depth of water in a given area?

Proposed Deliverables:

A short write up describing the project, results, and next steps or proposal to scale
Open source code for estimating flood depths from ground-based imagery
Example from flooding after a recent hurricane (e.g. Imelda, Florence, Harvey)
Descriptions of input data:

*Sources:

**Social media**

**News (TV, internet, etc.)**

Traffic cameras

Google Street View (“normal” imagery)


Input: Some image with water level

Output: Approximate the depth of the level base on the picture. 


### Ideas for data collections

- Since is from a corporation from Washington DC. Choose a City with histories of frequent flooding. 

- City Target: **New Orleans** for **Hurriance Katarina

- Find images before and during the flood.

- Find videos of hurricane to grab images?

- Find the actual flood depth information for Hurricane Katarine in New Orlean.


### Target 

- Object Dectection on the images.

- Train the model to recognize the object and give a height to those objects

- The flood depth will be the difference of the object that were not covered to its full height. 


### Resources

https://en.wikipedia.org/wiki/Louisiana_floods

https://en.wikipedia.org/wiki/List_of_Louisiana_hurricanes_(2000%E2%80%93present)

https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2-W5/5/2019/isprs-annals-IV-2-W5-5-2019.pdf

New Orlean Elevation: −6.5 to 20 ft (−2 to 6 m).*From wiki of New Orlean


