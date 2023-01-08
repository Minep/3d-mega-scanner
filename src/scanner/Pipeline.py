class PipelineStageBase:
    def __init__(self, name="pipeline1") -> None:
        self.myname = name

    def initialize(self, options):
        self._opts = options
    
    def execute(self, in_obj):
        raise NotImplementedError()
    
    def cleanup(self):
        pass

class Pipeline(PipelineStageBase):
    def __init__(self, config) -> None:
        self.stages = config
    
    def initialize(self, options):
        for stage in self.stages:
            print("INIT:", stage.myname)
            stage.initialize(options)
    
    def execute(self, in_obj):
        prev_in = in_obj
        for stages in self.stages:
            prev_in = stages.execute(prev_in)
        return prev_in

class PipelineOptions:
    def __init__(self, opts_dict) -> None:
        for k, v in opts_dict.items():
            if type(v) is dict:
                self.__dict__[k] = PipelineOptions(v)
            else:
                self.__dict__[k] = v