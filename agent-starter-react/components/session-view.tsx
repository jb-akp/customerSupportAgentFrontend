'use client';

import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  type AgentState,
  type ReceivedChatMessage,
  useRoomContext,
  useVoiceAssistant,
  useTranscriptions,
} from '@livekit/components-react';
import { toastAlert } from '@/components/alert-toast';
import { AgentControlBar } from '@/components/livekit/agent-control-bar/agent-control-bar';
import { ChatEntry } from '@/components/livekit/chat/chat-entry';
import { ChatMessageView } from '@/components/livekit/chat/chat-message-view';
import { MediaTiles } from '@/components/livekit/media-tiles';
import useChatAndTranscription from '@/hooks/useChatAndTranscription';
import { useDebugMode } from '@/hooks/useDebug';
import type { AppConfig } from '@/lib/types';
import { cn } from '@/lib/utils';

function isAgentAvailable(agentState: AgentState) {
  return agentState == 'listening' || agentState == 'thinking' || agentState == 'speaking';
}

interface SessionViewProps {
  appConfig: AppConfig;
  disabled: boolean;
  sessionStarted: boolean;
}

export const SessionView = ({
  appConfig,
  disabled,
  sessionStarted,
  ref,
}: React.ComponentProps<'div'> & SessionViewProps) => {
  const { state: agentState } = useVoiceAssistant();
  const [chatOpen, setChatOpen] = useState(false);
  const { messages, send } = useChatAndTranscription();
  const transcriptions = useTranscriptions();
  const room = useRoomContext();

  useDebugMode();

  async function handleSendMessage(message: string) {
    await send(message);
  }

  useEffect(() => {
    if (sessionStarted) {
      const timeout = setTimeout(() => {
        if (!isAgentAvailable(agentState)) {
          const reason =
            agentState === 'connecting'
              ? 'Agent did not join the room. '
              : 'Agent connected but did not complete initializing. ';

          toastAlert({
            title: 'Session ended',
            description: (
              <p className="w-full">
                {reason}
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href="https://docs.livekit.io/agents/start/voice-ai/"
                  className="whitespace-nowrap underline"
                >
                  See quickstart guide
                </a>
                .
              </p>
            ),
          });
          room.disconnect();
        }
      }, 10_000);

      return () => clearTimeout(timeout);
    }
  }, [agentState, sessionStarted, room]);

  const { supportsChatInput, supportsVideoInput, supportsScreenShare } = appConfig;
  const capabilities = {
    supportsChatInput,
    supportsVideoInput,
    supportsScreenShare,
  };

  return (
    <main
      ref={ref}
      inert={disabled}
      className={cn(
        'bg-slate-900 min-h-screen',
        !chatOpen && 'max-h-svh overflow-hidden'
      )}
    >
      {/* Header */}
      <div className="fixed top-0 left-0 right-0 z-40 bg-slate-800/80 backdrop-blur-md border-b border-slate-700">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M24 4C26.2091 4 28 5.79086 28 8C28 10.2091 26.2091 12 24 12C21.7909 12 20 10.2091 20 8C20 5.79086 21.7909 4 24 4Z" fill="white"/>
                <path d="M32 16C32 13.7909 30.2091 12 28 12H20C17.7909 12 16 13.7909 16 16V36C16 38.2091 17.7909 40 20 40H28C30.2091 40 32 38.2091 32 36V16Z" fill="white"/>
                <path d="M22 20C22 18.8954 22.8954 18 24 18C25.1046 18 26 18.8954 26 20V32C26 33.1046 25.1046 34 24 34C22.8954 34 22 33.1046 22 32V20Z" fill="#007AFF"/>
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-white">iPear Support</h1>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1 text-sm text-slate-400">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Connected</span>
            </div>
          </div>
        </div>
      </div>

      {/* Live Transcription Overlay */}
      {transcriptions.length > 0 && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-[60]">
          <div className="bg-black/60 text-white px-6 py-3 rounded-xl text-sm font-medium backdrop-blur-md border border-white/20 shadow-lg">
            {transcriptions[transcriptions.length - 1].text}
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <ChatMessageView
        className={cn(
          'mx-auto min-h-svh w-full max-w-2xl px-4 pt-24 pb-32 transition-[opacity,translate] duration-300 ease-out',
          chatOpen ? 'translate-y-0 opacity-100 delay-200' : 'translate-y-20 opacity-0'
        )}
      >
        <div className="space-y-4">
          <AnimatePresence>
            {messages.map((message: ReceivedChatMessage) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
              >
                <ChatEntry hideName key={message.id} entry={message} />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </ChatMessageView>

      {/* Video Area */}
      <div className="pt-16 pb-24">
        <MediaTiles chatOpen={chatOpen} />
      </div>

      {/* Control Bar */}
      <div className="fixed right-0 bottom-0 left-0 z-50 px-4 pb-12">
        <motion.div
          key="control-bar"
          initial={{ opacity: 0, translateY: '100%' }}
          animate={{
            opacity: sessionStarted ? 1 : 0,
            translateY: sessionStarted ? '0%' : '100%',
          }}
          transition={{ duration: 0.3, delay: sessionStarted ? 0.5 : 0, ease: 'easeOut' }}
        >
          <div className="relative z-10 mx-auto w-full max-w-2xl">
            {appConfig.isPreConnectBufferEnabled && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{
                  opacity: sessionStarted && messages.length === 0 ? 1 : 0,
                  transition: {
                    ease: 'easeIn',
                    delay: messages.length > 0 ? 0 : 0.8,
                    duration: messages.length > 0 ? 0.2 : 0.5,
                  },
                }}
                aria-hidden={messages.length > 0}
                className={cn(
                  'absolute inset-x-0 -top-12 text-center',
                  sessionStarted && messages.length === 0 && 'pointer-events-none'
                )}
              >
                <div className="inline-flex items-center space-x-2 bg-slate-800/90 backdrop-blur-sm px-4 py-2 rounded-full">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <p className="text-sm font-medium text-slate-300">
                    AI Assistant is listening
                  </p>
                </div>
              </motion.div>
            )}

            <div className="bg-slate-800/90 backdrop-blur-md rounded-2xl border border-slate-700 p-4">
              <AgentControlBar
                capabilities={capabilities}
                onChatOpenChange={setChatOpen}
                onSendMessage={handleSendMessage}
              />
            </div>
          </div>
        </motion.div>
      </div>
    </main>
  );
};
