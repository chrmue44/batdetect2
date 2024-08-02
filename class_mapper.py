from batdetect2.data.labels import ClassMapper
from soundevent import data
from typing import List, Optional

# The mapper is an object that gives a class to each sound event annotation.
# Annotations can have multiple tags but here you can define the class
# of the sound event using the tag info, or other info associated to the sound
# event if needed.
class Mapper(ClassMapper):

    # List of all possible classes the model can predict
    class_labels = [
        "social", 
        "feeding",
        "Barbastella barbastellus",
        "Eptesicus nilssonii",
        "Eptesicus serotinus",
        "Myotis daubentonii",
        "Myotis dasycneme",
        "Myotis emarginatus",
        "Myotis myotis",
        "Myotis mystacinus",
        "Myotis natteri",
        "Nyctalus leisleri",
        "Nyctalus noctula",
        "Pipistrellus kuhlii",
        "Pipistrellus nathusii",
        "Pipistrellus pipistrellus",
        "Pipistrellus pygmaeus",
        "Vespertilio murinus",
        "unknown"
    ]

    # Classify a given sound event annotation
    def encode(self, x: data.SoundEventAnnotation) -> Optional[str]:

        # Extract the "event" tag (e.g., "Echolocation" or "Social")
        event_tag = data.find_tag(x.tags, "event")

        # If it's a social call, return "social" as the class
        if event_tag.value == "Social":
            return "social"
        if event_tag.value == "Feeding":
            return "feeding"
        # If it's not an echolocation call, ignore it
        if event_tag.value != "Echolocation":
            return None

        # Extract the "class" tag (species) for echolocation calls
        species_tag = data.find_tag(x.tags, "class")
        return species_tag.value

    # Convert a class prediction back into annotation tags
    def decode(self, class_name: str) -> List[data.Tag]:
        if class_name == "social":
            return [data.Tag(key="event", value="social")]

        return [data.Tag(key="class", value=class_name)]        
