# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
from vllm.tokenizers.deepseek_v32 import DeepseekV32Tokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams


class DeepseekV32Renderer(BaseRenderer[DeepseekV32Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV32Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def _should_use_custom_template(self, params: ChatParams) -> bool:
        """Check if user provided a custom chat template."""
        return params.chat_template is not None

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format=(
                resolve_chat_template_content_format(
                    chat_template=params.chat_template,
                    tools=params.chat_template_kwargs.get("tools"),
                    given_format=params.chat_template_content_format,
                    tokenizer=self.get_tokenizer(),
                    model_config=self.model_config,
                )
                if self._should_use_custom_template(params)
                else "string"
            ),

            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        '''
        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )
        '''
        if self._should_use_custom_template(params):
            chat_template_kwargs = params.get_apply_chat_template_kwargs()
            prompt_raw = safe_apply_chat_template(
                self.model_config,
                self.get_tokenizer(),
                conversation,
                **chat_template_kwargs,
            )
        else:
            prompt_raw = self._apply_chat_template(
                conversation=conversation,
                messages=messages,
                **params.get_apply_chat_template_kwargs(),
            )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format=(
                resolve_chat_template_content_format(
                    chat_template=params.chat_template,
                    tools=params.chat_template_kwargs.get("tools"),
                    given_format=params.chat_template_content_format,
                    tokenizer=self.get_tokenizer(),
                    model_config=self.model_config,
                )
                if self._should_use_custom_template(params)
                else "string"
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        '''
        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )
        '''
        if self._should_use_custom_template(params):
            chat_template_kwargs = params.get_apply_chat_template_kwargs()
            prompt_raw = await make_async(
                safe_apply_chat_template, executor=self._executor
            )(
                self.model_config,
                self.get_tokenizer(),
                conversation,
                **chat_template_kwargs,
            )
        else:
            prompt_raw = await self._apply_chat_template_async(
                conversation=conversation,
                messages=messages,
                **params.get_apply_chat_template_kwargs(),
            )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
