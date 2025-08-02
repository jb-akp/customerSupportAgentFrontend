import * as React from 'react';
import type { MessageFormatter, ReceivedChatMessage } from '@livekit/components-react';
import { cn } from '@/lib/utils';
import { useChatMessage } from './hooks/utils';

export interface ChatEntryProps extends React.HTMLAttributes<HTMLLIElement> {
  /** The chat massage object to display. */
  entry: ReceivedChatMessage;
  /** Hide sender name. Useful when displaying multiple consecutive chat messages from the same person. */
  hideName?: boolean;
  /** Hide message timestamp. */
  hideTimestamp?: boolean;
  /** An optional formatter for the message body. */
  messageFormatter?: MessageFormatter;
}

export const ChatEntry = ({
  entry,
  messageFormatter,
  hideName,
  hideTimestamp,
  className,
  ...props
}: ChatEntryProps) => {
  const { message, hasBeenEdited, time, locale, name } = useChatMessage(entry, messageFormatter);

  const isUser = entry.from?.isLocal ?? false;
  const messageOrigin = isUser ? 'remote' : 'local';

  return (
    <li
      data-lk-message-origin={messageOrigin}
      title={time.toLocaleTimeString(locale, { timeStyle: 'full' })}
      className={cn('group flex flex-col gap-2', className)}
      {...props}
    >
      {(!hideTimestamp || !hideName || hasBeenEdited) && (
        <div className="flex items-center justify-between text-xs text-slate-400">
          {!hideName && (
            <span className="font-medium">
              {isUser ? 'You' : 'iPear Assistant'}
            </span>
          )}
          {!hideTimestamp && (
            <span className="font-mono opacity-0 transition-opacity ease-linear group-hover:opacity-100">
              {hasBeenEdited && '*'}
              {time.toLocaleTimeString(locale, { timeStyle: 'short' })}
            </span>
          )}
        </div>
      )}

      <div className={cn(
        'max-w-[85%] rounded-2xl px-4 py-3',
        isUser 
          ? 'bg-blue-500 text-white ml-auto' 
          : 'bg-slate-800 text-slate-100 mr-auto border border-slate-700'
      )}>
        <span className="text-sm leading-relaxed whitespace-pre-wrap">
          {message}
        </span>
      </div>
    </li>
  );
};
