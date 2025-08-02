'use client';

import * as React from 'react';
import { useCallback } from 'react';
import { Track } from 'livekit-client';
import { BarVisualizer, useRemoteParticipants } from '@livekit/components-react';
import { ChatTextIcon, PhoneDisconnectIcon } from '@phosphor-icons/react/dist/ssr';
import { ChatInput } from '@/components/livekit/chat/chat-input';
import { Button } from '@/components/ui/button';
import { Toggle } from '@/components/ui/toggle';
import { AppConfig } from '@/lib/types';
import { cn } from '@/lib/utils';
import { DeviceSelect } from '../device-select';
import { TrackToggle } from '../track-toggle';
import { UseAgentControlBarProps, useAgentControlBar } from './hooks/use-agent-control-bar';

export interface AgentControlBarProps
  extends React.HTMLAttributes<HTMLDivElement>,
    UseAgentControlBarProps {
  capabilities: Pick<AppConfig, 'supportsChatInput' | 'supportsVideoInput' | 'supportsScreenShare'>;
  onChatOpenChange?: (open: boolean) => void;
  onSendMessage?: (message: string) => Promise<void>;
  onDisconnect?: () => void;
  onDeviceError?: (error: { source: Track.Source; error: Error }) => void;
}

/**
 * A control bar specifically designed for voice assistant interfaces
 */
export function AgentControlBar({
  controls,
  saveUserChoices = true,
  capabilities,
  className,
  onSendMessage,
  onChatOpenChange,
  onDisconnect,
  onDeviceError,
  ...props
}: AgentControlBarProps) {
  const participants = useRemoteParticipants();
  const [chatOpen, setChatOpen] = React.useState(false);
  const [isSendingMessage, setIsSendingMessage] = React.useState(false);

  const isAgentAvailable = participants.some((p) => p.isAgent);
  const isInputDisabled = !chatOpen || !isAgentAvailable || isSendingMessage;

  const [isDisconnecting, setIsDisconnecting] = React.useState(false);

  const {
    micTrackRef,
    visibleControls,
    cameraToggle,
    microphoneToggle,
    screenShareToggle,
    handleAudioDeviceChange,
    handleVideoDeviceChange,
    handleDisconnect,
  } = useAgentControlBar({
    controls,
    saveUserChoices,
  });

  const handleSendMessage = async (message: string) => {
    setIsSendingMessage(true);
    try {
      await onSendMessage?.(message);
    } finally {
      setIsSendingMessage(false);
    }
  };

  const onLeave = async () => {
    setIsDisconnecting(true);
    await handleDisconnect();
    setIsDisconnecting(false);
    onDisconnect?.();
  };

  React.useEffect(() => {
    onChatOpenChange?.(chatOpen);
  }, [chatOpen, onChatOpenChange]);

  const onMicrophoneDeviceSelectError = useCallback(
    (error: Error) => {
      onDeviceError?.({ source: Track.Source.Microphone, error });
    },
    [onDeviceError]
  );
  const onCameraDeviceSelectError = useCallback(
    (error: Error) => {
      onDeviceError?.({ source: Track.Source.Camera, error });
    },
    [onDeviceError]
  );

  return (
    <div
      aria-label="Voice assistant controls"
      className={cn(
        'flex flex-col gap-4',
        className
      )}
      {...props}
    >
      {capabilities.supportsChatInput && (
        <div
          inert={!chatOpen}
          className={cn(
            'overflow-hidden transition-[height] duration-300 ease-out',
            chatOpen ? 'h-[57px]' : 'h-0'
          )}
        >
          <div className="flex h-8 w-full">
            <ChatInput onSend={handleSendMessage} disabled={isInputDisabled} className="w-full" />
          </div>
        </div>
      )}

      <div className="flex flex-row justify-between items-center gap-3">
        <div className="flex gap-2">
          {visibleControls.microphone && (
            <div className="flex items-center gap-1">
              <TrackToggle
                variant="primary"
                source={Track.Source.Microphone}
                pressed={microphoneToggle.enabled}
                disabled={microphoneToggle.pending}
                onPressedChange={microphoneToggle.toggle}
                className="peer/track group/track relative w-auto pr-3 pl-3 rounded-xl bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors"
              >
                <BarVisualizer
                  barCount={3}
                  trackRef={micTrackRef}
                  options={{ minHeight: 5 }}
                  className="flex h-full w-auto items-center justify-center gap-0.5"
                >
                  <span
                    className={cn([
                      'h-full w-0.5 origin-center rounded-2xl',
                      'group-data-[state=on]/track:bg-blue-500 group-data-[state=off]/track:bg-red-500',
                      'data-lk-muted:bg-slate-400',
                    ])}
                  ></span>
                </BarVisualizer>
              </TrackToggle>
              <DeviceSelect
                size="sm"
                kind="audioinput"
                onMediaDeviceError={onMicrophoneDeviceSelectError}
                onActiveDeviceChange={handleAudioDeviceChange}
                className={cn([
                  'pl-2 text-sm',
                  'peer-data-[state=off]/track:text-red-500',
                  'hover:text-slate-300 focus:text-slate-300',
                  'hover:peer-data-[state=off]/track:text-red-500 focus:peer-data-[state=off]/track:text-red-500',
                  'hidden md:block',
                ])}
              />
            </div>
          )}

          {capabilities.supportsVideoInput && visibleControls.camera && (
            <div className="flex items-center gap-1">
              <TrackToggle
                variant="primary"
                source={Track.Source.Camera}
                pressed={cameraToggle.enabled}
                pending={cameraToggle.pending}
                disabled={cameraToggle.pending}
                onPressedChange={cameraToggle.toggle}
                className="peer/track relative w-auto rounded-xl pr-3 pl-3 bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors"
              />
              <DeviceSelect
                size="sm"
                kind="videoinput"
                onMediaDeviceError={onCameraDeviceSelectError}
                onActiveDeviceChange={handleVideoDeviceChange}
                className={cn([
                  'pl-2 text-sm',
                  'peer-data-[state=off]/track:text-red-500',
                  'hover:text-slate-300 focus:text-slate-300',
                  'hover:peer-data-[state=off]/track:text-red-500 focus:peer-data-[state=off]/track:text-red-500',
                ])}
              />
            </div>
          )}

          {capabilities.supportsScreenShare && visibleControls.screenShare && (
            <div className="flex items-center gap-1">
              <TrackToggle
                variant="secondary"
                source={Track.Source.ScreenShare}
                pressed={screenShareToggle.enabled}
                disabled={screenShareToggle.pending}
                onPressedChange={screenShareToggle.toggle}
                className="relative w-auto rounded-xl bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors"
              />
            </div>
          )}

          {/* {visibleControls.chat && (
            <Toggle
              variant="secondary"
              aria-label="Toggle chat"
              pressed={chatOpen}
              onPressedChange={setChatOpen}
              disabled={!isAgentAvailable}
              className="aspect-square h-full rounded-xl bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors"
            >
              <ChatTextIcon weight="bold" />
            </Toggle>
          )} */}
        </div>
        {visibleControls.leave && (
          <Button
            variant="destructive"
            onClick={onLeave}
            disabled={isDisconnecting}
            className="font-medium rounded-xl bg-red-500 hover:bg-red-600 text-white px-4 py-2 transition-colors"
          >
            <PhoneDisconnectIcon weight="bold" />
            <span className="hidden md:inline ml-2">End Call</span>
            <span className="inline md:hidden ml-2">End</span>
          </Button>
        )}
      </div>
    </div>
  );
}
