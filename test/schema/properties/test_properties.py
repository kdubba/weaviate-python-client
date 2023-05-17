import unittest
from unittest.mock import Mock

from requests.exceptions import ConnectionError as RequestsConnectionError

from test.util import mock_connection_func, check_error_message, check_startswith_error_message
from weaviate.exceptions import (
    UnexpectedStatusCodeException,
    SchemaValidationException,
)
from weaviate.schema.properties import CrudProperty


class TestCRUDProperty(unittest.TestCase):
    def test_create(self):
        """
        Test `create` method.
        """

        prop = CrudProperty(Mock())

        # invalid calls
        error_message = "Class name must be of type str but is "
        check_property_error_message = 'Property does not contain "dataType"'
        requests_error_message = "Property was created properly."

        with self.assertRaises(TypeError) as error:
            prop.create(35, {})
        check_error_message(self, error, error_message + str(int))

        # test if `check_property` is called in `create`
        with self.assertRaises(SchemaValidationException) as error:
            prop.create("Class", {})
        check_error_message(self, error, check_property_error_message)

        prop = CrudProperty(
            mock_connection_func("post", side_effect=RequestsConnectionError("Test!"))
        )
        with self.assertRaises(RequestsConnectionError) as error:
            prop.create("Class", {"name": "test", "dataType": ["test_type"]})
        check_error_message(self, error, requests_error_message)

        prop = CrudProperty(mock_connection_func("post", status_code=404))
        with self.assertRaises(UnexpectedStatusCodeException) as error:
            prop.create("Class", {"name": "test", "dataType": ["test_type"]})
        check_startswith_error_message(self, error, "Add property to class")

        # valid calls
        connection_mock = mock_connection_func("post")  # Mock calling weaviate
        prop = CrudProperty(connection_mock)

        test_prop = {
            "dataType": ["string"],
            "description": "my Property",
            "moduleConfig": {"text2vec-contextionary": {"vectorizePropertyName": True}},
            "name": "superProp",
            "indexInverted": True,
        }

        prop.create("TestThing", test_prop)

        connection_mock.post.assert_called_with(
            path="/schema/TestThing/properties",
            weaviate_object=test_prop,
        )

        prop.create("testThing", test_prop)

        connection_mock.post.assert_called_with(
            path="/schema/TestThing/properties",
            weaviate_object=test_prop,
        )
