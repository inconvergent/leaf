about leaf.py
=============

implementation of leaf venation pattern simulation based on this paper 
(http://algorithmicbotany.org/papers/venation.sig2005.html) by Adam Runions et al.

the code is tested with the following python (2.7.3) modules:  
		-		numpy version 1.7.0 (\*)  
		-		scipy version '0.13.0.dev-4f297fc' (need 0.12 or higher for the incremental version of Delaunay triangulation to be included.)  
		-		cairo 1.8.8 (\*)  

(\*) most newer versions should work

note:
-----
there are a number of things that don't work quite as they should yet. the
most notable one being the merging of branches. i hope to improve this soon.

license:
--------
you can do as you please with this code, as long as you don't make money from
it.


----
Anders Hoff 2013

