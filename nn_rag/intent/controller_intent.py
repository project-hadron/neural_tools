import inspect
from nn_rag.components.commons import Commons
from ds_core.intent.abstract_intent import AbstractIntentModel
from nn_rag.managers.controller_property_manager import ControllerPropertyManager


class ControllerIntentModel(AbstractIntentModel):

    """This components provides a set of actions that focuses on the Controller. The Controller is a unique components
    that independently orchestrates the components registered to it. It executes the components Domain Contract and
    not its code. The Controller orchestrates how those components should run with the components being independent
    in their actions and therefore a separation of concerns."""

    def __init__(self, property_manager: ControllerPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'base'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical']
        intent_type_additions = []
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, run_level: str, source: str=None, persist: [str, list]=None,
                            controller_repo: str=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters.

        :param run_level:
        :param persist:
        :param source:
        :param controller_repo: (optional) the controller repo to use if no uri_pm_repo is within the intent parameters
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return: Canonical with parameterised intent applied
        """
        # get the list of levels to run
        if not self._pm.has_intent(run_level):
            raise ValueError(f"The intent level '{run_level}' could not be found in the "
                             f"property manager '{self._pm.manager_name()}' for task '{self._pm.task_name}'")
        shape = None
        level_key = self._pm.join(self._pm.KEY.intent_key, run_level)
        for order in sorted(self._pm.get(level_key, {})):
            for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                if method in self.__dir__():
                    # failsafe in case kwargs was stored as the reference
                    params.update(params.pop('kwargs', {}))
                    # add method kwargs to the params
                    if isinstance(kwargs, dict):
                        params.update(kwargs)
                    # remove the creator param
                    _ = params.pop('intent_creator', 'Unknown')
                    # add excluded params and set to False
                    params.update({'save_intent': False})
                    # add the controller_repo if given
                    if isinstance(controller_repo, str) and 'uri_pm_repo' not in params.keys():
                        params.update({'uri_pm_repo': controller_repo})
                    shape = eval(f"self.{method}(source=source, persist=persist, **{params})", globals(), locals())
        return shape

