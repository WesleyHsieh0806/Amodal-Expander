# Generate sim-link to each dataset
Segment="/compute/trinity-1-38/chengyeh/TAO-Amodal-Segment-Object-Large"
TAO_Amodal="/compute/trinity-1-38/chengyeh/TAO-Amodal"

ln -s $Segment segment_object_large
ln -s $TAO_Amodal tao_amodal