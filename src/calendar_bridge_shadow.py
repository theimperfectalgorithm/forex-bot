"""Explicit one-shot shadow observation. No production modules or config imports."""
import argparse
import logging

from core.calendar_bridge import BridgeReader, ExpectedIdentity, ShadowReporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--terminal-path', required=True)
    parser.add_argument('--terminal-data-path', required=True)
    parser.add_argument('--login', required=True)
    parser.add_argument('--server', required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--company')
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--existing-state', required=True, choices=['CLEAR', 'BLACKOUT', 'UNKNOWN'],
                        help='Caller-supplied Task018 observation; this command does not refresh news')
    args = parser.parse_args()
    identity = ExpectedIdentity(args.login, args.server, args.terminal_path,
                                args.terminal_data_path, args.instance_id, args.company)
    logging.basicConfig(level=logging.INFO)
    reader = BridgeReader(args.directory, identity)
    ShadowReporter().compare(reader, args.symbol, args.existing_state, logging.getLogger('bridge-shadow'))


if __name__ == '__main__':
    main()
