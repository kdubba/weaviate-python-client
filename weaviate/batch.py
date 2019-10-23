import validators


class ReferenceBatchRequest:

    def __init__(self):
        self.from_thing_class_names = []
        self.from_thing_ids = []
        self.from_thing_properties = []
        self.to_thing_ids = []

    def add_reference(self, from_thing_class_name, from_thing_uuid, from_thing_property, to_thing_uuid):

        if not isinstance(from_thing_class_name, str) or not isinstance(from_thing_uuid, str) or \
                not isinstance(from_thing_property, str) or not isinstance(to_thing_uuid, str):
            raise ValueError('All arguments must be of type string')

        self.from_thing_class_names.append(from_thing_class_name)
        self.from_thing_ids.append(from_thing_uuid)
        self.from_thing_properties.append(from_thing_property)
        self.to_thing_ids.append(to_thing_uuid)

    def get_batch_size(self):
        return len(self.from_thing_class_names)

    def get_request_body(self):
        batch_body = [None] * self.get_batch_size()
        for i in range(self.get_batch_size()):
            batch_body[i] = {
                "from": "weaviate://localhost/things/" + self.from_thing_class_names[i] + "/" + self.from_thing_ids[i] + "/" +
                        self.from_thing_properties[i],
                "to": "weaviate://localhost/things/" + self.to_thing_ids[i]
            }
        return batch_body


class ThingsBatchRequest:
    """
    Collect things for one batch request to weaviate.
    Caution this batch will not be validated through weaviate.
    """

    def __init__(self):
        self.things = []

    def add_thing(self, thing, class_name, uuid=None):
        """ Add a thing to this batch

        :param thing: that should be added as part of the batch
        :type thing: dict
        :param class_name: class of the thing
        :type class_name: str
        :param uuid: if given the thing will be added under this uuid
        :type uuid: str
        :raises TypeError, ValueError
        """
        if not isinstance(thing, dict):
            raise TypeError
        if not isinstance(class_name, str):
            raise TypeError

        batch_item = {
            "class": class_name,
            "schema": thing
        }
        if uuid is not None:
            if not isinstance(uuid, str):
                raise TypeError
            if not validators.uuid(uuid):
                raise ValueError
            batch_item["id"] = uuid

        self.things.append(batch_item)

    def get_request_body(self):
        """ Get the request body as it is needed for weaviate

        :return: the request body as a dict
        """
        return {
            "fields": [
                "ALL"
            ],
            "things": self.things
        }