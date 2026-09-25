from learning.intermimic_players_distill import InterMimicPlayerContinuousDistill


class InterMimicPlayerContinuousStudentWM(InterMimicPlayerContinuousDistill):
    def run(self):
        super().run()
        try:
            save_fn = self.env.task.save_recorded_wm_videos
        except AttributeError:
            save_fn = None
        if save_fn is not None:
            save_fn()
        return

    def set_full_state_weights(self, weights):
        # Base player checkpoints are primarily model weights; optimizer/epoch
        # fields are training-only. Keep this lightweight for evaluation.
        self.set_weights(weights)
        try:
            adapter = self.env.task.world_model_adapter
        except AttributeError:
            adapter = None
        if adapter is not None and "student_world_model" in weights:
            adapter.load_state_dict(weights["student_world_model"], load_optimizer=False)
